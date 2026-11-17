#include <Arduino.h>

// ============================================================
// ESP32 CENTERING V11
// Dual independent PI RPM closed-loop control
//
// Compatible with Jetson commands:
//   DRIVE,speed,turn
//   MANUAL,F|B,speed,steering
//   STOP
//   HELLO
//
// turn > 0 = LEFT
// turn < 0 = RIGHT
//
// No D control.
// Each wheel has its own PI speed loop.
// ============================================================


// ============================================================
// L298N MOTOR PINS
// ============================================================

// LEFT MOTOR
const int ENA = 25;
const int IN1 = 26;
const int IN2 = 27;

// RIGHT MOTOR
const int ENB = 33;
const int IN3 = 32;
const int IN4 = 14;


// ============================================================
// ENCODER PINS
// ============================================================

// LEFT
const int LEFT_ENC_A = 18;
const int LEFT_ENC_B = 19;

// RIGHT
const int RIGHT_ENC_A = 22;
const int RIGHT_ENC_B = 23;


// ============================================================
// ENCODER COUNTS PER REVOLUTION
// ============================================================

const float LEFT_COUNTS_PER_REV  = 143.0f;
const float RIGHT_COUNTS_PER_REV = 147.0f;


// ============================================================
// RPM CONTROL SETTINGS
//
// Start with conservative values.
// These can be tuned later if needed.
// ============================================================

float KP_LEFT  = 0.45f;
float KI_LEFT  = 0.08f;

float KP_RIGHT = 0.45f;
float KI_RIGHT = 0.08f;


// PI integral anti-windup
const float INTEGRAL_LIMIT = 180.0f;


// RPM control loop period
const unsigned long RPM_INTERVAL = 50;


// ============================================================
// SPEED -> TARGET RPM
//
// Jetson AUTO normally uses 0.12 ~ 0.24.
// Manual uses 0.30.
//
// 0.12 -> 105 RPM
// 0.24 -> 155 RPM
// >=0.24 -> 155 RPM
// ============================================================

const float JETSON_MIN_SPEED = 0.12f;
const float JETSON_MAX_SPEED = 0.24f;

const float MIN_TARGET_RPM = 105.0f;
const float MAX_TARGET_RPM = 155.0f;


// ============================================================
// STEERING TARGET RPM DIFFERENCE
//
// Positive turn = LEFT:
//   left target RPM decreases
//   right target RPM increases
//
// At |turn| = 1.0, each wheel target changes by 35% of base RPM.
// The total left/right target difference is 70% of base RPM.
//
// Example base=140 RPM, turn=+0.5:
// delta = 140 * 0.35 * 0.5 = 24.5
// left  = 115.5 RPM
// right = 164.5 RPM
// ============================================================

const float MAX_TURN_RATIO = 0.35f;

// Straight-line calibration: the car drifts left, so the right wheel
// needs a slightly lower target speed. Keep this active in both modes.
const float RIGHT_WHEEL_RPM_TRIM = 0.95f;


// Keep target RPM from becoming too low while moving
const float MIN_MOVING_TARGET_RPM = 75.0f;


// ============================================================
// PWM SETTINGS
//
// Feed-forward gives the PI controller a useful starting point.
// PI feedback then corrects each wheel independently.
//
// Left/right feed-forward can be different, but PI is responsible
// for removing the actual RPM difference.
// ============================================================

const int LEFT_PWM_AT_MIN_RPM  = 110;
const int RIGHT_PWM_AT_MIN_RPM = 110;

const int LEFT_PWM_AT_MAX_RPM  = 140;
const int RIGHT_PWM_AT_MAX_RPM = 140;


// Minimum usable PWM on the floor
const int LEFT_MIN_PWM  = 100;
const int RIGHT_MIN_PWM = 100;


// Maximum allowed PWM
const int MAX_PWM = 200;


// ============================================================
// STARTUP BOOST
//
// Helps overcome static friction when starting from STOP.
// ============================================================

const int LEFT_START_PWM  = 145;
const int RIGHT_START_PWM = 140;

const unsigned long START_BOOST_TIME = 100;
const int START_BOOST_STEER_PWM = 28;  // Allow immediate steering during boost.


// ============================================================
// SERIAL / SAFETY
// ============================================================

const unsigned long SERIAL_BAUD = 115200;

const unsigned long COMMAND_TIMEOUT = 500;

const unsigned long STATUS_INTERVAL = 200;


// ============================================================
// ENCODER COUNTERS
// ============================================================

volatile long leftEncoderCount = 0;
volatile long rightEncoderCount = 0;

long lastLeftEncoderCount = 0;
long lastRightEncoderCount = 0;


// ============================================================
// MEASURED RPM
// ============================================================

float measuredLeftRPM = 0.0f;
float measuredRightRPM = 0.0f;


// ============================================================
// TARGET RPM
// ============================================================

float targetLeftRPM = 0.0f;
float targetRightRPM = 0.0f;


// ============================================================
// PI STATES
// ============================================================

float errorLeft = 0.0f;
float errorRight = 0.0f;

float integralLeft = 0.0f;
float integralRight = 0.0f;

float piLeft = 0.0f;
float piRight = 0.0f;


// ============================================================
// VEHICLE STATE
//
// mode:
// 0 = STOP
// 1 = AUTO
// 2 = MANUAL
//
// direction:
// 1  = FORWARD
// -1 = BACKWARD
// 0  = STOP
// ============================================================

int currentMode = 0;
int currentDirection = 0;

float currentSpeed = 0.0f;
float currentTurn = 0.0f;

int currentManualSteering = 0;


// ============================================================
// CURRENT PWM
// ============================================================

int currentLeftPWM = 0;
int currentRightPWM = 0;


// ============================================================
// START BOOST STATE
// ============================================================

bool startBoostActive = false;
unsigned long startBoostStarted = 0;


// ============================================================
// TIMERS
// ============================================================

unsigned long lastCommandTime = 0;
unsigned long lastRPMTime = 0;
unsigned long lastStatusTime = 0;


// ============================================================
// SERIAL BUFFER
// ============================================================

char serialBuffer[100];
int serialIndex = 0;

// Only the newest complete non-safety command is executed.
// Older queued DRIVE/MANUAL commands are overwritten.
char latestCommand[100];
bool latestCommandReady = false;


// ============================================================
// ENCODER ISR
// ============================================================

void IRAM_ATTR leftEncoderISR()
{
  leftEncoderCount++;
}

void IRAM_ATTR rightEncoderISR()
{
  rightEncoderCount++;
}


// ============================================================
// MOTOR DIRECTION
// ============================================================

void setMotorDirection(int direction)
{
  if (direction > 0)
  {
    // FORWARD
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);

    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
  }
  else if (direction < 0)
  {
    // BACKWARD
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);

    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
  }
  else
  {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);

    digitalWrite(IN3, LOW);
    digitalWrite(IN4, LOW);
  }
}


// ============================================================
// PWM OUTPUT
// ============================================================

void outputMotorPWM(int leftPWM, int rightPWM)
{
  leftPWM = constrain(leftPWM, 0, MAX_PWM);
  rightPWM = constrain(rightPWM, 0, MAX_PWM);

  currentLeftPWM = leftPWM;
  currentRightPWM = rightPWM;

  analogWrite(ENA, leftPWM);
  analogWrite(ENB, rightPWM);
}


// ============================================================
// RESET PI
// ============================================================

void resetPI()
{
  errorLeft = 0.0f;
  errorRight = 0.0f;

  integralLeft = 0.0f;
  integralRight = 0.0f;

  piLeft = 0.0f;
  piRight = 0.0f;
}


// ============================================================
// STOP MOTORS
// ============================================================

void stopMotors()
{
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);

  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);

  currentMode = 0;
  currentDirection = 0;

  currentSpeed = 0.0f;
  currentTurn = 0.0f;
  currentManualSteering = 0;

  targetLeftRPM = 0.0f;
  targetRightRPM = 0.0f;

  currentLeftPWM = 0;
  currentRightPWM = 0;

  startBoostActive = false;

  resetPI();
}


// ============================================================
// STARTUP BOOST
// ============================================================

void beginStartBoost()
{
  startBoostActive = true;
  startBoostStarted = millis();

  // Do not carry old integral into a new start.
  resetPI();
}

bool isStartBoostActive()
{
  if (!startBoostActive)
    return false;

  if (millis() - startBoostStarted >= START_BOOST_TIME)
  {
    startBoostActive = false;

    // Start PI cleanly after boost.
    resetPI();

    return false;
  }

  return true;
}


// ============================================================
// SPEED -> BASE TARGET RPM
// ============================================================

float speedToBaseRPM(float speedValue)
{
  if (speedValue <= 0.0f)
    return 0.0f;

  float ratio =
    (speedValue - JETSON_MIN_SPEED) /
    (JETSON_MAX_SPEED - JETSON_MIN_SPEED);

  ratio =
    constrain(
      ratio,
      0.0f,
      1.0f
    );

  return
    MIN_TARGET_RPM +
    ratio *
    (MAX_TARGET_RPM - MIN_TARGET_RPM);
}


// ============================================================
// TARGET RPM FROM SPEED + TURN
// ============================================================

void calculateTargetRPM(float speedValue, float turn)
{
  float baseRPM =
    speedToBaseRPM(speedValue);

  if (baseRPM <= 0.0f)
  {
    targetLeftRPM = 0.0f;
    targetRightRPM = 0.0f;
    return;
  }

  turn =
    constrain(
      turn,
      -1.0f,
      1.0f
    );

  float delta =
    baseRPM *
    MAX_TURN_RATIO *
    turn;

  // Positive turn = LEFT:
  // left slower, right faster
  targetLeftRPM =
    baseRPM - delta;

  targetRightRPM =
    (baseRPM + delta) * RIGHT_WHEEL_RPM_TRIM;

  targetLeftRPM =
    constrain(
      targetLeftRPM,
      MIN_MOVING_TARGET_RPM,
      MAX_TARGET_RPM * 1.35f
    );

  targetRightRPM =
    constrain(
      targetRightRPM,
      MIN_MOVING_TARGET_RPM,
      MAX_TARGET_RPM * 1.35f
    );
}


// ============================================================
// TARGET RPM -> FEED-FORWARD PWM
// ============================================================

int feedForwardPWM(
  float targetRPM,
  bool leftWheel
)
{
  if (targetRPM <= 0.0f)
    return 0;

  float ratio =
    (targetRPM - MIN_TARGET_RPM) /
    (MAX_TARGET_RPM - MIN_TARGET_RPM);

  ratio =
    constrain(
      ratio,
      0.0f,
      1.0f
    );

  int pwm;

  if (leftWheel)
  {
    pwm =
      LEFT_PWM_AT_MIN_RPM +
      (int)round(
        ratio *
        (LEFT_PWM_AT_MAX_RPM - LEFT_PWM_AT_MIN_RPM)
      );

    pwm =
      constrain(
        pwm,
        LEFT_MIN_PWM,
        MAX_PWM
      );
  }
  else
  {
    pwm =
      RIGHT_PWM_AT_MIN_RPM +
      (int)round(
        ratio *
        (RIGHT_PWM_AT_MAX_RPM - RIGHT_PWM_AT_MIN_RPM)
      );

    pwm =
      constrain(
        pwm,
        RIGHT_MIN_PWM,
        MAX_PWM
      );
  }

  return pwm;
}


// ============================================================
// PI RPM CONTROL
// ============================================================

void updatePIControl(float dt)
{
  if (
    currentMode == 0 ||
    currentDirection == 0 ||
    targetLeftRPM <= 0.0f ||
    targetRightRPM <= 0.0f
  )
  {
    return;
  }

  // During startup boost, do not integrate.
  if (isStartBoostActive())
  {
    outputMotorPWM(
      LEFT_START_PWM,
      RIGHT_START_PWM
    );

    return;
  }

  // ----------------------------------------------------------
  // Errors
  // ----------------------------------------------------------

  errorLeft =
    targetLeftRPM -
    measuredLeftRPM;

  errorRight =
    targetRightRPM -
    measuredRightRPM;


  // ----------------------------------------------------------
  // Integral
  // ----------------------------------------------------------

  integralLeft +=
    errorLeft *
    dt;

  integralRight +=
    errorRight *
    dt;


  integralLeft =
    constrain(
      integralLeft,
      -INTEGRAL_LIMIT,
      INTEGRAL_LIMIT
    );

  integralRight =
    constrain(
      integralRight,
      -INTEGRAL_LIMIT,
      INTEGRAL_LIMIT
    );


  // ----------------------------------------------------------
  // PI
  // ----------------------------------------------------------

  piLeft =
    KP_LEFT *
    errorLeft +
    KI_LEFT *
    integralLeft;

  piRight =
    KP_RIGHT *
    errorRight +
    KI_RIGHT *
    integralRight;


  // ----------------------------------------------------------
  // Feed-forward + PI correction
  // ----------------------------------------------------------

  int leftPWM =
    feedForwardPWM(
      targetLeftRPM,
      true
    )
    +
    (int)round(piLeft);


  int rightPWM =
    feedForwardPWM(
      targetRightRPM,
      false
    )
    +
    (int)round(piRight);


  // ----------------------------------------------------------
  // Keep motors inside usable range
  // ----------------------------------------------------------

  leftPWM =
    constrain(
      leftPWM,
      LEFT_MIN_PWM,
      MAX_PWM
    );

  rightPWM =
    constrain(
      rightPWM,
      RIGHT_MIN_PWM,
      MAX_PWM
    );


  outputMotorPWM(
    leftPWM,
    rightPWM
  );
}


// ============================================================
// LOW-LATENCY IMMEDIATE MOTOR OUTPUT
//
// Target RPM changes from a new Jetson command are reflected in PWM
// immediately using feed-forward + the most recent PI correction.
// The 50 ms PI loop then refines the actual wheel RPM.
// ============================================================

void applyImmediateMotorOutput()
{
  if (
    currentMode == 0 ||
    currentDirection == 0 ||
    currentSpeed <= 0.0f
  )
  {
    return;
  }

  setMotorDirection(currentDirection);

  // During startup boost, do NOT ignore steering.
  if (isStartBoostActive())
  {
    float effectiveTurn = currentTurn;

    // Reverse manual steering needs opposite wheel differential
    // to feel intuitive at the vehicle body.
    if (currentDirection < 0)
    {
      effectiveTurn = -effectiveTurn;
    }

    int boostDelta =
      (int)round(
        effectiveTurn *
        START_BOOST_STEER_PWM
      );

    int leftPWM =
      LEFT_START_PWM -
      boostDelta;

    int rightPWM =
      RIGHT_START_PWM +
      boostDelta;

    leftPWM =
      constrain(
        leftPWM,
        LEFT_MIN_PWM,
        MAX_PWM
      );

    rightPWM =
      constrain(
        rightPWM,
        RIGHT_MIN_PWM,
        MAX_PWM
      );

    outputMotorPWM(
      leftPWM,
      rightPWM
    );

    return;
  }

  int leftPWM =
    feedForwardPWM(
      targetLeftRPM,
      true
    )
    +
    (int)round(piLeft);

  int rightPWM =
    feedForwardPWM(
      targetRightRPM,
      false
    )
    +
    (int)round(piRight);

  leftPWM =
    constrain(
      leftPWM,
      LEFT_MIN_PWM,
      MAX_PWM
    );

  rightPWM =
    constrain(
      rightPWM,
      RIGHT_MIN_PWM,
      MAX_PWM
    );

  outputMotorPWM(
    leftPWM,
    rightPWM
  );
}


// ============================================================
// SERIAL RECEIVE - LATEST COMMAND WINS
//
// If the UART contains:
//   DRIVE old1
//   DRIVE old2
//   DRIVE newest
//
// old1/old2 are discarded and only newest is executed.
//
// STOP remains safety-priority and is executed immediately.
// ============================================================

void receiveLatestSerialCommand()
{
  while (Serial.available() > 0)
  {
    char incoming =
      Serial.read();

    if (
      incoming == '\n' ||
      incoming == '\r'
    )
    {
      if (serialIndex > 0)
      {
        serialBuffer[serialIndex] = '\0';

        // Safety STOP must never wait behind another command.
        if (strcmp(serialBuffer, "STOP") == 0)
        {
          latestCommandReady = false;
          processCommand(serialBuffer);
        }
        else
        {
          // Overwrite any older waiting command.
          strncpy(
            latestCommand,
            serialBuffer,
            sizeof(latestCommand) - 1
          );

          latestCommand[
            sizeof(latestCommand) - 1
          ] = '\0';

          latestCommandReady = true;
        }

        serialIndex = 0;
      }
    }
    else
    {
      if (
        serialIndex <
        sizeof(serialBuffer) - 1
      )
      {
        serialBuffer[serialIndex] =
          incoming;

        serialIndex++;
      }
      else
      {
        // Invalid overlong command: drop this partial line.
        serialIndex = 0;
      }
    }
  }

  // UART has been drained. Execute only the newest complete command.
  if (latestCommandReady)
  {
    latestCommandReady = false;
    processCommand(latestCommand);
  }
}


// ============================================================
// AUTO COMMAND
// ============================================================

void driveAuto(
  float speedValue,
  float turn
)
{
  if (speedValue <= 0.0f)
  {
    stopMotors();
    return;
  }

  bool wasStopped =
    currentMode == 0;

  bool directionChanged =
    currentDirection != 0 &&
    currentDirection != 1;

  currentMode = 1;
  currentDirection = 1;

  currentSpeed = speedValue;
  currentTurn =
    constrain(
      turn,
      -1.0f,
      1.0f
    );

  currentManualSteering = 0;

  setMotorDirection(1);

  calculateTargetRPM(
    currentSpeed,
    currentTurn
  );

  if (wasStopped || directionChanged)
  {
    beginStartBoost();
  }

  // Apply the newest target immediately instead of waiting
  // for the next 50 ms RPM control tick.
  applyImmediateMotorOutput();
}


// ============================================================
// MANUAL COMMAND
// ============================================================

void driveManual(
  int direction,
  float speedValue,
  int steering
)
{
  if (
    direction == 0 ||
    speedValue <= 0.0f
  )
  {
    stopMotors();
    return;
  }

  bool wasStopped =
    currentMode == 0;

  bool directionChanged =
    currentDirection != 0 &&
    currentDirection != direction;

  currentMode = 2;
  currentDirection = direction;
  currentSpeed = speedValue;
  currentManualSteering = steering;

  // Convert MANUAL steering +/-35 into normalized turn +/-1.
  float manualTurn =
    constrain(
      steering / 35.0f,
      -1.0f,
      1.0f
    );

  currentTurn = manualTurn;

  setMotorDirection(
    currentDirection
  );

  calculateTargetRPM(
    currentSpeed,
    currentTurn
  );

  if (wasStopped || directionChanged)
  {
    beginStartBoost();
  }

  // Apply the newest target immediately instead of waiting
  // for the next 50 ms RPM control tick.
  applyImmediateMotorOutput();
}


// ============================================================
// RPM UPDATE
// ============================================================

void updateRPM()
{
  unsigned long now =
    millis();

  unsigned long elapsed =
    now -
    lastRPMTime;

  if (elapsed < RPM_INTERVAL)
    return;

  long leftNow;
  long rightNow;

  noInterrupts();

  leftNow =
    leftEncoderCount;

  rightNow =
    rightEncoderCount;

  interrupts();

  long deltaLeft =
    leftNow -
    lastLeftEncoderCount;

  long deltaRight =
    rightNow -
    lastRightEncoderCount;

  lastLeftEncoderCount =
    leftNow;

  lastRightEncoderCount =
    rightNow;

  float dt =
    elapsed /
    1000.0f;


  // ==========================================================
  // Calculate RPM magnitude
  // ==========================================================

  measuredLeftRPM =
    (
      (float)deltaLeft /
      LEFT_COUNTS_PER_REV
    )
    /
    dt
    *
    60.0f;

  measuredRightRPM =
    (
      (float)deltaRight /
      RIGHT_COUNTS_PER_REV
    )
    /
    dt
    *
    60.0f;


  // ==========================================================
  // Run PI
  // ==========================================================

  updatePIControl(dt);


  // ==========================================================
  // STATUS
  // ==========================================================

  if (
    now -
    lastStatusTime >=
    STATUS_INTERVAL
  )
  {
    Serial.print("STATUS");

    Serial.print(" MODE=");
    Serial.print(currentMode);

    Serial.print(" DIR=");
    Serial.print(currentDirection);

    Serial.print(" SPD=");
    Serial.print(currentSpeed, 3);

    Serial.print(" TURN=");
    Serial.print(currentTurn, 3);

    Serial.print(" TGT_L=");
    Serial.print(targetLeftRPM, 1);

    Serial.print(" TGT_R=");
    Serial.print(targetRightRPM, 1);

    Serial.print(" RPM_L=");
    Serial.print(measuredLeftRPM, 1);

    Serial.print(" RPM_R=");
    Serial.print(measuredRightRPM, 1);

    Serial.print(" ERR_L=");
    Serial.print(errorLeft, 1);

    Serial.print(" ERR_R=");
    Serial.print(errorRight, 1);

    Serial.print(" I_L=");
    Serial.print(integralLeft, 1);

    Serial.print(" I_R=");
    Serial.print(integralRight, 1);

    Serial.print(" PI_L=");
    Serial.print(piLeft, 1);

    Serial.print(" PI_R=");
    Serial.print(piRight, 1);

    Serial.print(" PWM_L=");
    Serial.print(currentLeftPWM);

    Serial.print(" PWM_R=");
    Serial.print(currentRightPWM);

    Serial.print(" BOOST=");
    Serial.print(
      startBoostActive ? 1 : 0
    );

    Serial.println();

    lastStatusTime =
      now;
  }

  lastRPMTime =
    now;
}


// ============================================================
// PROCESS SERIAL COMMAND
// ============================================================

void processCommand(char *command)
{
  // ==========================================================
  // HELLO
  // ==========================================================

  if (strcmp(command, "HELLO") == 0)
  {
    // Jetson expects this exact string.
    Serial.println(
      "READY CENTERING V10"
    );

    return;
  }


  // ==========================================================
  // STOP
  // ==========================================================

  if (strcmp(command, "STOP") == 0)
  {
    lastCommandTime =
      millis();

    stopMotors();

    return;
  }


  // ==========================================================
  // CTRL compatibility
  //
  // Jetson sends:
  // CTRL,0.000,999
  // when opening serial.
  // ==========================================================

  if (strncmp(command, "CTRL,", 5) == 0)
  {
    char *data =
      command + 5;

    char *speedText =
      strtok(
        data,
        ","
      );

    char *steeringText =
      strtok(
        NULL,
        ","
      );

    if (
      speedText == NULL ||
      steeringText == NULL
    )
    {
      return;
    }

    float speedValue =
      atof(speedText);

    int steering =
      atoi(steeringText);

    lastCommandTime =
      millis();

    if (
      speedValue <= 0.0f ||
      steering == 999
    )
    {
      stopMotors();
    }

    return;
  }


  // ==========================================================
  // AUTO
  //
  // DRIVE,0.120,0.0000
  // ==========================================================

  if (strncmp(command, "DRIVE,", 6) == 0)
  {
    char *data =
      command + 6;

    char *speedText =
      strtok(
        data,
        ","
      );

    char *turnText =
      strtok(
        NULL,
        ","
      );

    if (
      speedText == NULL ||
      turnText == NULL
    )
    {
      return;
    }

    float speedValue =
      atof(speedText);

    float turn =
      atof(turnText);

    lastCommandTime =
      millis();

    driveAuto(
      speedValue,
      turn
    );

    return;
  }


  // ==========================================================
  // MANUAL
  //
  // MANUAL,F,0.300,0
  // MANUAL,B,0.300,0
  // MANUAL,F,0.300,35
  // MANUAL,F,0.300,-35
  // ==========================================================

  if (strncmp(command, "MANUAL,", 7) == 0)
  {
    char *data =
      command + 7;

    char *directionText =
      strtok(
        data,
        ","
      );

    char *speedText =
      strtok(
        NULL,
        ","
      );

    char *steeringText =
      strtok(
        NULL,
        ","
      );

    if (
      directionText == NULL ||
      speedText == NULL ||
      steeringText == NULL
    )
    {
      return;
    }

    int direction = 0;

    if (directionText[0] == 'F')
    {
      direction = 1;
    }
    else if (directionText[0] == 'B')
    {
      direction = -1;
    }

    float speedValue =
      atof(speedText);

    int steering =
      atoi(steeringText);

    lastCommandTime =
      millis();

    driveManual(
      direction,
      speedValue,
      steering
    );

    return;
  }
}


// ============================================================
// SETUP
// ============================================================

void setup()
{
  Serial.begin(
    SERIAL_BAUD
  );

  delay(1000);


  // ==========================================================
  // MOTOR PINS
  // ==========================================================

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);


  // ==========================================================
  // ENCODER PINS
  // ==========================================================

  pinMode(
    LEFT_ENC_A,
    INPUT_PULLUP
  );

  pinMode(
    LEFT_ENC_B,
    INPUT_PULLUP
  );

  pinMode(
    RIGHT_ENC_A,
    INPUT_PULLUP
  );

  pinMode(
    RIGHT_ENC_B,
    INPUT_PULLUP
  );


  attachInterrupt(
    digitalPinToInterrupt(
      LEFT_ENC_A
    ),
    leftEncoderISR,
    RISING
  );


  attachInterrupt(
    digitalPinToInterrupt(
      RIGHT_ENC_A
    ),
    rightEncoderISR,
    RISING
  );


  // ==========================================================
  // PWM
  // ==========================================================

  analogWriteResolution(
    ENA,
    8
  );

  analogWriteResolution(
    ENB,
    8
  );

  analogWriteFrequency(
    ENA,
    1000
  );

  analogWriteFrequency(
    ENB,
    1000
  );


  // ==========================================================
  // INITIAL STOP
  // ==========================================================

  stopMotors();


  lastCommandTime =
    millis();

  lastRPMTime =
    millis();

  lastStatusTime =
    millis();


  // IMPORTANT:
  // Do not print BOOT / alternate READY messages because
  // the Jetson program treats them as disconnected.
  Serial.println(
    "READY CENTERING V10"
  );
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
  // ==========================================================
  // SERIAL RECEIVE
  //
  // Drain UART and execute only the newest complete command.
  // ==========================================================

  receiveLatestSerialCommand();


  // ==========================================================
  // SAFETY TIMEOUT
  // ==========================================================

  if (
    currentMode != 0 &&
    millis() -
    lastCommandTime >
    COMMAND_TIMEOUT
  )
  {
    stopMotors();
  }


  // ==========================================================
  // ENCODER + PI CONTROL
  // ==========================================================

  updateRPM();
}
