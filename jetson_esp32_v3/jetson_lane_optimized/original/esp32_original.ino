// ============================================================
// ESP32 + L298N + Dual Encoder
// Stable Lane Control Version
// ============================================================


// ============================================================
// MOTOR
// ============================================================

const int ENA = 25;
const int IN1 = 26;
const int IN2 = 27;

const int ENB = 33;
const int IN3 = 32;
const int IN4 = 14;


// ============================================================
// ENCODER
// ============================================================

const int LEFT_ENC_A = 18;
const int LEFT_ENC_B = 19;

const int RIGHT_ENC_A = 22;
const int RIGHT_ENC_B = 23;

const float LEFT_COUNTS_PER_REV = 143.0f;
const float RIGHT_COUNTS_PER_REV = 147.0f;


// ============================================================
// BASE PWM
//
// 你實測：左 115 / 右 110
// ============================================================

const int LEFT_BASE_PWM = 115;
const int RIGHT_BASE_PWM = 110;

const float CALIBRATED_SPEED = 0.30f;


// ============================================================
// PWM LIMIT
// ============================================================

const int LEFT_MIN_PWM = 100;
const int RIGHT_MIN_PWM = 95;

const int MAX_PWM = 200;


// ============================================================
// STARTUP KICK
// ============================================================

const bool ENABLE_STARTUP_KICK = true;

const int LEFT_KICK_PWM = 135;
const int RIGHT_KICK_PWM = 130;

const unsigned long STARTUP_KICK_TIME = 150;


// ============================================================
// STEERING
//
// 這次降低差速強度
// ============================================================

const float STEERING_GAIN = 0.18f;

const int MAX_STEERING_CORRECTION = 10;


// ============================================================
// WHEEL SYNC
// ============================================================

const bool ENABLE_WHEEL_SYNC = true;

const float SYNC_DEADZONE_RPM = 4.0f;

const float SYNC_ADAPT_GAIN = 0.02f;

const float MAX_SYNC_TRIM = 12.0f;

float syncTrim = 0.0f;


// ============================================================
// SERIAL
// ============================================================

const unsigned long SERIAL_BAUD = 115200;

const unsigned long COMMAND_TIMEOUT = 1000;


// ============================================================
// RPM
// ============================================================

const unsigned long RPM_INTERVAL = 500;

unsigned long lastCommandTime = 0;
unsigned long lastRPMTime = 0;


// ============================================================
// COUNTS
// ============================================================

volatile long leftEncoderCount = 0;
volatile long rightEncoderCount = 0;

long lastLeftEncoderCount = 0;
long lastRightEncoderCount = 0;

float measuredLeftRPM = 0.0f;
float measuredRightRPM = 0.0f;


// ============================================================
// STATE
// ============================================================

float currentSpeed = 0.0f;

int currentSteering = 0;
int currentDirection = 0;

int currentLeftPWM = 0;
int currentRightPWM = 0;

bool motorsWereStopped = true;

String serialBuffer = "";


// ============================================================
// ISR
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
// DIRECTION
// ============================================================

void setMotorDirection(int direction)
{
  if (direction > 0)
  {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);

    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
  }

  else if (direction < 0)
  {
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
// PWM
// ============================================================

void writeMotorPWM(
  int leftPWM,
  int rightPWM
)
{
  leftPWM = constrain(
    leftPWM,
    0,
    255
  );

  rightPWM = constrain(
    rightPWM,
    0,
    255
  );

  currentLeftPWM = leftPWM;
  currentRightPWM = rightPWM;

  analogWrite(
    ENA,
    leftPWM
  );

  analogWrite(
    ENB,
    rightPWM
  );
}


// ============================================================
// STOP
// ============================================================

void stopMotors()
{
  currentSpeed = 0.0f;
  currentSteering = 0;
  currentDirection = 0;

  currentLeftPWM = 0;
  currentRightPWM = 0;

  analogWrite(
    ENA,
    0
  );

  analogWrite(
    ENB,
    0
  );

  setMotorDirection(0);

  motorsWereStopped = true;
}


// ============================================================
// BASE PWM
// ============================================================

void calculateBasePWM(
  float speed,
  int &leftPWM,
  int &rightPWM
)
{
  if (speed <= 0.0f)
  {
    leftPWM = 0;
    rightPWM = 0;
    return;
  }

  float ratio =
    speed /
    CALIBRATED_SPEED;

  leftPWM =
    (int)(
      LEFT_BASE_PWM *
      ratio
    );

  rightPWM =
    (int)(
      RIGHT_BASE_PWM *
      ratio
    );

  leftPWM = constrain(
    leftPWM,
    LEFT_MIN_PWM,
    MAX_PWM
  );

  rightPWM = constrain(
    rightPWM,
    RIGHT_MIN_PWM,
    MAX_PWM
  );
}


// ============================================================
// START KICK
// ============================================================

void startupKick(
  int direction
)
{
  if (
    !ENABLE_STARTUP_KICK
    ||
    !motorsWereStopped
  )
  {
    return;
  }

  setMotorDirection(
    direction
  );

  analogWrite(
    ENA,
    LEFT_KICK_PWM
  );

  analogWrite(
    ENB,
    RIGHT_KICK_PWM
  );

  delay(
    STARTUP_KICK_TIME
  );
}


// ============================================================
// SYNC
// ============================================================

void applyWheelSync(
  int &leftPWM,
  int &rightPWM
)
{
  if (!ENABLE_WHEEL_SYNC)
    return;

  if (
    currentDirection != 1
    ||
    currentSteering != 0
    ||
    currentSpeed <= 0.0f
  )
  {
    return;
  }

  int leftCorrection =
    (int)(
      syncTrim /
      2.0f
    );

  int rightCorrection =
    (int)(
      syncTrim -
      leftCorrection
    );

  leftPWM -=
    leftCorrection;

  rightPWM +=
    rightCorrection;

  leftPWM = constrain(
    leftPWM,
    LEFT_MIN_PWM,
    MAX_PWM
  );

  rightPWM = constrain(
    rightPWM,
    RIGHT_MIN_PWM,
    MAX_PWM
  );
}


// ============================================================
// DRIVE
// ============================================================

void driveVehicle(
  float speed,
  int steering,
  int direction
)
{
  if (
    speed <= 0.0f
    ||
    direction == 0
  )
  {
    stopMotors();
    return;
  }

  bool starting =
    motorsWereStopped;

  currentSpeed = speed;
  currentSteering = steering;
  currentDirection = direction;

  int leftPWM = 0;
  int rightPWM = 0;

  calculateBasePWM(
    speed,
    leftPWM,
    rightPWM
  );

  // ========================================================
  // GENTLE DIFFERENTIAL STEERING
  // ========================================================

  if (steering != 0)
  {
    int correction =
      (int)(
        abs(steering)
        *
        STEERING_GAIN
      );

    correction = constrain(
      correction,
      0,
      MAX_STEERING_CORRECTION
    );

    if (steering > 0)
    {
      // LEFT
      leftPWM -= correction;
      rightPWM += correction;
    }

    else
    {
      // RIGHT
      leftPWM += correction;
      rightPWM -= correction;
    }

    leftPWM = constrain(
      leftPWM,
      LEFT_MIN_PWM,
      MAX_PWM
    );

    rightPWM = constrain(
      rightPWM,
      RIGHT_MIN_PWM,
      MAX_PWM
    );
  }

  applyWheelSync(
    leftPWM,
    rightPWM
  );

  if (starting)
  {
    startupKick(
      direction
    );
  }

  setMotorDirection(
    direction
  );

  writeMotorPWM(
    leftPWM,
    rightPWM
  );

  motorsWereStopped = false;
}


// ============================================================
// MANUAL LEFT
// ============================================================

void manualLeft(float speed)
{
  int leftPWM = 0;
  int rightPWM = 0;

  calculateBasePWM(
    speed,
    leftPWM,
    rightPWM
  );

  currentSpeed = speed;
  currentSteering = 35;
  currentDirection = 1;

  setMotorDirection(1);

  writeMotorPWM(
    0,
    rightPWM
  );

  motorsWereStopped = false;
}


// ============================================================
// MANUAL RIGHT
// ============================================================

void manualRight(float speed)
{
  int leftPWM = 0;
  int rightPWM = 0;

  calculateBasePWM(
    speed,
    leftPWM,
    rightPWM
  );

  currentSpeed = speed;
  currentSteering = -35;
  currentDirection = 1;

  setMotorDirection(1);

  writeMotorPWM(
    leftPWM,
    0
  );

  motorsWereStopped = false;
}


// ============================================================
// UPDATE SYNC
// ============================================================

void updateWheelSync()
{
  if (!ENABLE_WHEEL_SYNC)
    return;

  if (
    currentDirection != 1
    ||
    currentSteering != 0
    ||
    currentSpeed <= 0.0f
  )
  {
    return;
  }

  if (
    measuredLeftRPM < 30.0f
    ||
    measuredRightRPM < 30.0f
  )
  {
    return;
  }

  float rpmError =
    measuredLeftRPM -
    measuredRightRPM;

  if (
    abs(rpmError)
    <=
    SYNC_DEADZONE_RPM
  )
  {
    return;
  }

  syncTrim +=
    rpmError *
    SYNC_ADAPT_GAIN;

  syncTrim = constrain(
    syncTrim,
    -MAX_SYNC_TRIM,
    MAX_SYNC_TRIM
  );

  int leftPWM = 0;
  int rightPWM = 0;

  calculateBasePWM(
    currentSpeed,
    leftPWM,
    rightPWM
  );

  applyWheelSync(
    leftPWM,
    rightPWM
  );

  writeMotorPWM(
    leftPWM,
    rightPWM
  );
}


// ============================================================
// RPM
// ============================================================

void updateRPM()
{
  unsigned long now =
    millis();

  if (
    now -
    lastRPMTime
    <
    RPM_INTERVAL
  )
  {
    return;
  }

  float dt =
    (
      now -
      lastRPMTime
    )
    /
    1000.0f;

  noInterrupts();

  long leftCount =
    leftEncoderCount;

  long rightCount =
    rightEncoderCount;

  interrupts();

  long deltaLeft =
    leftCount -
    lastLeftEncoderCount;

  long deltaRight =
    rightCount -
    lastRightEncoderCount;

  lastLeftEncoderCount =
    leftCount;

  lastRightEncoderCount =
    rightCount;

  measuredLeftRPM =
    (
      (
        (float)deltaLeft
        /
        LEFT_COUNTS_PER_REV
      )
      /
      dt
    )
    *
    60.0f;

  measuredRightRPM =
    (
      (
        (float)deltaRight
        /
        RIGHT_COUNTS_PER_REV
      )
      /
      dt
    )
    *
    60.0f;

  float rpmError =
    measuredLeftRPM -
    measuredRightRPM;

  updateWheelSync();

  Serial.print(
    "STATUS SPD="
  );

  Serial.print(
    currentSpeed,
    3
  );

  Serial.print(
    " STEER="
  );

  Serial.print(
    currentSteering
  );

  Serial.print(
    " PWM_L="
  );

  Serial.print(
    currentLeftPWM
  );

  Serial.print(
    " PWM_R="
  );

  Serial.print(
    currentRightPWM
  );

  Serial.print(
    " ENC_L="
  );

  Serial.print(
    deltaLeft
  );

  Serial.print(
    " RPM_L="
  );

  Serial.print(
    measuredLeftRPM,
    2
  );

  Serial.print(
    " ENC_R="
  );

  Serial.print(
    deltaRight
  );

  Serial.print(
    " RPM_R="
  );

  Serial.print(
    measuredRightRPM,
    2
  );

  Serial.print(
    " ERR="
  );

  Serial.print(
    rpmError,
    2
  );

  Serial.print(
    " SYNC="
  );

  Serial.println(
    syncTrim,
    2
  );

  lastRPMTime = now;
}


// ============================================================
// AUTO COMMAND
// ============================================================

void processCTRL(String line)
{
  int c1 =
    line.indexOf(',');

  int c2 =
    line.indexOf(
      ',',
      c1 + 1
    );

  if (
    c1 < 0
    ||
    c2 < 0
  )
  {
    return;
  }

  float speed =
    line.substring(
      c1 + 1,
      c2
    ).toFloat();

  int steering =
    line.substring(
      c2 + 1
    ).toInt();

  if (
    speed <= 0.0f
    ||
    steering == 999
  )
  {
    stopMotors();
    return;
  }

  steering = constrain(
    steering,
    -45,
    45
  );

  driveVehicle(
    speed,
    steering,
    1
  );
}


// ============================================================
// MANUAL COMMAND
// ============================================================

void processMANUAL(String line)
{
  int c1 =
    line.indexOf(',');

  int c2 =
    line.indexOf(
      ',',
      c1 + 1
    );

  int c3 =
    line.indexOf(
      ',',
      c2 + 1
    );

  if (
    c1 < 0
    ||
    c2 < 0
    ||
    c3 < 0
  )
  {
    return;
  }

  String direction =
    line.substring(
      c1 + 1,
      c2
    );

  float speed =
    line.substring(
      c2 + 1,
      c3
    ).toFloat();

  int steering =
    line.substring(
      c3 + 1
    ).toInt();

  if (direction == "F")
  {
    if (steering >= 30)
    {
      manualLeft(
        speed
      );
    }

    else if (steering <= -30)
    {
      manualRight(
        speed
      );
    }

    else
    {
      driveVehicle(
        speed,
        0,
        1
      );
    }
  }

  else if (direction == "B")
  {
    driveVehicle(
      speed,
      0,
      -1
    );
  }

  else
  {
    stopMotors();
  }
}


// ============================================================
// COMMAND PARSER
// ============================================================

void processCommand(String line)
{
  line.trim();

  if (
    line.length()
    ==
    0
  )
  {
    return;
  }

  lastCommandTime =
    millis();

  if (
    line.startsWith(
      "CTRL,"
    )
  )
  {
    processCTRL(
      line
    );
  }

  else if (
    line.startsWith(
      "MANUAL,"
    )
  )
  {
    processMANUAL(
      line
    );
  }
}


// ============================================================
// SERIAL
// ============================================================

void readSerial()
{
  while (
    Serial.available()
  )
  {
    char c =
      Serial.read();

    if (c == '\n')
    {
      processCommand(
        serialBuffer
      );

      serialBuffer = "";
    }

    else if (c != '\r')
    {
      serialBuffer += c;

      if (
        serialBuffer.length()
        >
        120
      )
      {
        serialBuffer = "";
      }
    }
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

  pinMode(
    ENA,
    OUTPUT
  );

  pinMode(
    IN1,
    OUTPUT
  );

  pinMode(
    IN2,
    OUTPUT
  );

  pinMode(
    ENB,
    OUTPUT
  );

  pinMode(
    IN3,
    OUTPUT
  );

  pinMode(
    IN4,
    OUTPUT
  );

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

  stopMotors();

  lastCommandTime =
    millis();

  lastRPMTime =
    millis();

  Serial.println(
    "READY ESP32 STABLE CONTROL"
  );
}


// ============================================================
// LOOP
// ============================================================

void loop()
{
  readSerial();

  updateRPM();

  if (
    currentSpeed > 0.0f
    &&
    millis() -
    lastCommandTime
    >
    COMMAND_TIMEOUT
  )
  {
    stopMotors();

    Serial.println(
      "EMERGENCY COMMAND TIMEOUT"
    );
  }

  delay(1);
}