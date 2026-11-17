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
const float MIN_TURN_RATIO = 0.95f; // Maximum inside-wheel reduction is 5%.
const float AUTO_STEERING_LIMIT = 15.0f; // Matches Jetson MAX_STEERING = 15.

// Non-blocking starting pulse, followed by low-speed control.
// If the car cannot overcome static friction, calibrate minimum PWM first.
const bool ENABLE_STARTUP_KICK = true;
const int LEFT_KICK_PWM = 130, RIGHT_KICK_PWM = 130;
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
  if (speed <= 0 || direction == 0) { stopMotors(); return; }
  bool starting = currentDirection == 0;
  currentSpeed = speed;
  currentSteering = steering;
  currentDirection = direction;
  manualMode = manual;
  if (appliedDirection != 0 && appliedDirection != direction) {
    writePWM(0, 0);
    setMotorDirection(0);
    reversing = true;
    reverseStarted = millis();
    kickActive = false;
    syncTrim = 0;
  }
  if (starting && !reversing && ENABLE_STARTUP_KICK) {
    kickActive = true;
    kickStarted = millis();
  }
  // No delay or motor writes here: control loop owns normal PWM output.
}

void updateMotors() {
  uint32_t now = millis();
  if (currentSpeed > 0 && (uint32_t)(now - lastCommandTime) >= COMMAND_TIMEOUT) {
    stopMotors();
    Serial.println("EMERGENCY COMMAND TIMEOUT");
  }
  if ((uint32_t)(now - lastControlTime) < CONTROL_INTERVAL) return;
  float dt = fminf((uint32_t)(now - lastControlTime) / 1000.0f, 0.05f);
  lastControlTime = now;
  if (currentDirection == 0) return;
  if (reversing) {
    if ((uint32_t)(now - reverseStarted) < REVERSE_DEADTIME) return;
    reversing = false;
  }
  setMotorDirection(currentDirection);
  float left, right;
  calculateBasePWM(currentSpeed, left, right);
  if (manualMode && currentDirection > 0 && abs(currentSteering) >= 30) {
    if (currentSteering > 0) left = 0; else right = 0;
  } else {
    // Approaching the target reduces Jetson steering; the ratio then approaches 1.
    float turnStrength = bound(fabsf(currentSteering) / AUTO_STEERING_LIMIT, 0, 1);
    float turnRatio = 1.0f - (1.0f - MIN_TURN_RATIO) * turnStrength;
    if (currentSteering > 0) {
      left *= turnRatio;  // Positive steering requests LEFT: right PWM unchanged.
    } else if (currentSteering < 0) {
      right *= turnRatio; // Negative steering requests RIGHT: left PWM unchanged.
    } else if (ENABLE_WHEEL_SYNC && currentDirection == 1) {
      left -= syncTrim * 0.5f;
      right += syncTrim * 0.5f;
    }
    // Preserve proportional reduction instead of clamping it back to minimum.
    left = bound(left, 0, MAX_PWM);
    right = bound(right, 0, MAX_PWM);
  }
  if (kickActive) {
    if ((uint32_t)(now - kickStarted) < STARTUP_KICK_TIME) {
      if (left > 0) left = fmaxf(left, LEFT_KICK_PWM);
      if (right > 0) right = fmaxf(right, RIGHT_KICK_PWM);
      syncWindowEligible = false;
      writePWM(left, right);
      return; // Bypass slew limit for kick; STOP still handled every loop.
    } else kickActive = false;
  }
  float nextLeft = approachPWM(actualLeftPWM, left, dt);
  float nextRight = approachPWM(actualRightPWM, right, dt);
  bool settledStraight = currentDirection == 1 && currentSteering == 0 && !kickActive
                         && fabsf(nextLeft - left) < 1 && fabsf(nextRight - right) < 1;
  if (!settledStraight) syncWindowEligible = false;
  writePWM(nextLeft, nextRight);
}

void updateRPM() {
  uint32_t now = millis();
  if ((uint32_t)(now - lastRPMTime) < RPM_INTERVAL) return;
  float dt = (uint32_t)(now - lastRPMTime) / 1000.0f;
  // Aligned 32-bit reads on ESP32; counters are only written by their own ISR.
  uint32_t left = leftEncoderCount, right = rightEncoderCount;
  uint32_t dl = left - lastLeftCount, dr = right - lastRightCount;
  lastLeftCount = left; lastRightCount = right; lastRPMTime = now;
  measuredLeftRPM = dl * 60.0f / (LEFT_COUNTS_PER_REV * dt);
  measuredRightRPM = dr * 60.0f / (RIGHT_COUNTS_PER_REV * dt);
  if (ENABLE_WHEEL_SYNC && syncWindowEligible && currentDirection == 1 && currentSteering == 0
      && measuredLeftRPM >= 30 && measuredRightRPM >= 30) {
    float error = measuredLeftRPM - measuredRightRPM;
    if (fabsf(error) > SYNC_DEADZONE_RPM)
      syncTrim = bound(syncTrim + error * SYNC_ADAPT_GAIN, -MAX_SYNC_TRIM, MAX_SYNC_TRIM);
  }
  syncWindowEligible = currentDirection == 1 && currentSteering == 0 && !reversing && !kickActive;
  Serial.print("STATUS SPD="); Serial.print(currentSpeed, 3);
  Serial.print(" STEER="); Serial.print(currentSteering);
  Serial.print(" PWM_L="); Serial.print(actualLeftPWM, 1);
  Serial.print(" PWM_R="); Serial.print(actualRightPWM, 1);
  Serial.print(" RPM_L="); Serial.print(measuredLeftRPM, 2);
  Serial.print(" RPM_R="); Serial.print(measuredRightRPM, 2);
  Serial.print(" SYNC="); Serial.println(syncTrim, 2);
}

bool parseFloat(const char *s, float &value) {
  char *end;
  value = strtof(s, &end);
  return end != s && *end == '\0' && isfinite(value);
}

bool parseSteering(const char *s, int &value) {
  char *end;
  long parsed = strtol(s, &end, 10);
  if (end == s || *end != '\0' || (parsed != 999 && (parsed < -45 || parsed > 45))) return false;
  value = (int)parsed;
  return true;
}

void processCommand(char *line) {
  // Split without accepting empty fields or ignored trailing garbage.
  char *fields[5];
  int count = 1;
  fields[0] = line;
  for (char *p = line; *p; ++p) {
    if (*p == ',') {
      if (count >= 5) return;
      *p = '\0'; fields[count++] = p + 1;
    }
  }
  bool manual = strcmp(fields[0], "MANUAL") == 0;
  if ((!manual && (strcmp(fields[0], "CTRL") != 0 || count != 3)) || (manual && count != 4)) return;
  float speed;
  int steering, direction = 1;
  if (!parseFloat(fields[manual ? 2 : 1], speed) || speed < 0 || speed > CALIBRATED_SPEED) return;
  if (!parseSteering(fields[manual ? 3 : 2], steering)) return;
  if (manual) {
    if (strcmp(fields[1], "F") == 0) direction = 1;
    else if (strcmp(fields[1], "B") == 0) direction = -1;
    else return;
  }
  lastCommandTime = millis(); // Only a valid complete command renews watchdog.
  if (speed == 0 || steering == 999) stopMotors();
  else driveVehicle(speed, steering, direction, manual);
}

void readSerial() {
  int budget = 64; // Keep control/watchdog running even during a serial flood.
  while (budget-- > 0 && Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      if (!discardLine) { serialBuffer[serialLength] = '\0'; processCommand(serialBuffer); }
      serialLength = 0; discardLine = false;
    } else if (c != '\r' && !discardLine) {
      if (c == '\0' || serialLength >= sizeof(serialBuffer) - 1) {
        discardLine = true; serialLength = 0;
      } else serialBuffer[serialLength++] = c;
    }
  }
}

void setup() {
  Serial.begin(115200);
  for (int pin : {ENA, IN1, IN2, ENB, IN3, IN4}) pinMode(pin, OUTPUT);
  for (int pin : {LEFT_ENC_A, LEFT_ENC_B, RIGHT_ENC_A, RIGHT_ENC_B}) pinMode(pin, INPUT_PULLUP);
  stopMotors();
  attachInterrupt(digitalPinToInterrupt(LEFT_ENC_A), leftEncoderISR, RISING);
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENC_A), rightEncoderISR, RISING);
  lastCommandTime = lastControlTime = lastRPMTime = millis();
  Serial.println("READY ESP32 ADAPTIVE TURN 0.95 TO 1.00");
}

void loop() {
  readSerial();
  updateMotors();
  updateRPM();
  delay(1);
}
