// Jetson v3 companion. Same wiring, baud, steering sign and A-RISING encoder CPR.
// Not an RPM PI controller: RPM calibration is still required before adding PI.
#include <Arduino.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <initializer_list>

const int ENA = 25, IN1 = 26, IN2 = 27;
const int ENB = 33, IN3 = 32, IN4 = 14;
const int LEFT_ENC_A = 18, LEFT_ENC_B = 19;
const int RIGHT_ENC_A = 22, RIGHT_ENC_B = 23;
const float LEFT_COUNTS_PER_REV = 143.0f, RIGHT_COUNTS_PER_REV = 147.0f;
const int LEFT_BASE_PWM = 115, RIGHT_BASE_PWM = 110;
const int LEFT_MIN_PWM = 100, RIGHT_MIN_PWM = 95;
const int MAX_PWM = 200;
const float CALIBRATED_SPEED = 0.30f;
const float START_SPEED = 0.12f;
const float STEERING_GAIN = 0.18f;
const int MAX_STEERING_CORRECTION = 10;

// Default OFF: the old immediate 135/130 pulse caused abrupt starts.
// If the car cannot overcome static friction, calibrate minimum PWM first.
const bool ENABLE_STARTUP_KICK = false;
const int LEFT_KICK_PWM = 135, RIGHT_KICK_PWM = 130;
const uint32_t STARTUP_KICK_TIME = 150;
const float PWM_RISE_PER_SECOND = 100.0f;
const float PWM_FALL_PER_SECOND = 240.0f;
const uint32_t CONTROL_INTERVAL = 20;
const uint32_t REVERSE_DEADTIME = 120;
const uint32_t COMMAND_TIMEOUT = 500;
const uint32_t RPM_INTERVAL = 500;

const bool ENABLE_WHEEL_SYNC = true;
const float SYNC_DEADZONE_RPM = 4.0f;
const float SYNC_ADAPT_GAIN = 0.02f, MAX_SYNC_TRIM = 12.0f;
float syncTrim = 0.0f;
volatile uint32_t leftEncoderCount = 0, rightEncoderCount = 0;
uint32_t lastLeftCount = 0, lastRightCount = 0;
float measuredLeftRPM = 0, measuredRightRPM = 0;
float currentSpeed = 0, actualLeftPWM = 0, actualRightPWM = 0;
int currentSteering = 0, currentDirection = 0, appliedDirection = 0;
bool manualMode = false, reversing = false, kickActive = false;
bool syncWindowEligible = false;
uint32_t lastCommandTime = 0, lastControlTime = 0, lastRPMTime = 0;
uint32_t reverseStarted = 0, kickStarted = 0;
char serialBuffer[96];
size_t serialLength = 0;
bool discardLine = false;

void IRAM_ATTR leftEncoderISR() { ++leftEncoderCount; }
void IRAM_ATTR rightEncoderISR() { ++rightEncoderCount; }

float bound(float x, float lo, float hi) { return fminf(hi, fmaxf(lo, x)); }

void setMotorDirection(int direction) {
  digitalWrite(IN1, direction > 0 ? HIGH : LOW);
  digitalWrite(IN2, direction < 0 ? HIGH : LOW);
  digitalWrite(IN3, direction > 0 ? HIGH : LOW);
  digitalWrite(IN4, direction < 0 ? HIGH : LOW);
  appliedDirection = direction;
}

void writePWM(float left, float right) {
  actualLeftPWM = bound(left, 0, MAX_PWM);
  actualRightPWM = bound(right, 0, MAX_PWM);
  analogWrite(ENA, (int)lroundf(actualLeftPWM));
  analogWrite(ENB, (int)lroundf(actualRightPWM));
}

void stopMotors() {
  currentSpeed = 0;
  currentSteering = 0;
  currentDirection = 0;
  writePWM(0, 0);               // STOP bypasses every ramp and kick
  setMotorDirection(0);
  reversing = kickActive = false;
  syncWindowEligible = false;
  syncTrim = 0;
}

void calculateBasePWM(float speed, float &left, float &right) {
  if (speed <= 0) { left = right = 0; return; }
  // Piecewise calibration: .12 => original running minimum, .30 => original base.
  // This preserves measured settings while eliminating the old .12-.24 plateau.
  float fraction = bound((speed - START_SPEED) / (CALIBRATED_SPEED - START_SPEED), 0, 1);
  left = LEFT_MIN_PWM + fraction * (LEFT_BASE_PWM - LEFT_MIN_PWM);
  right = RIGHT_MIN_PWM + fraction * (RIGHT_BASE_PWM - RIGHT_MIN_PWM);
}

float approachPWM(float previous, float target, float dt) {
  float limit = (target > previous ? PWM_RISE_PER_SECOND : PWM_FALL_PER_SECOND) * dt;
  return previous + bound(target - previous, -limit, limit);
}

void driveVehicle(float speed, int steering, int direction, bool manual) {