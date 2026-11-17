#include <Arduino.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <initializer_list>

// Same wiring as the original vehicle.
const int ENA=25, IN1=26, IN2=27, ENB=33, IN3=32, IN4=14;
const int LEFT_ENC_A=18, LEFT_ENC_B=19, RIGHT_ENC_A=22, RIGHT_ENC_B=23;
const float LEFT_COUNTS_PER_REV=143.0f, RIGHT_COUNTS_PER_REV=147.0f;
// Left feedforward reduced another 2 PWM after the latest straight-driving test.
const int LEFT_MIN_PWM=154, RIGHT_MIN_PWM=166;
const int LEFT_BASE_PWM=168, RIGHT_BASE_PWM=180;
const int LEFT_KICK_PWM=168, RIGHT_KICK_PWM=180;
const int MAX_PWM=200;
const float START_SPEED=0.12f, CALIBRATED_SPEED=0.30f;
const float MAX_TURN_REDUCTION=0.03f; // Jetson requests 0%-3% inside-wheel reduction.
const float MAX_TOTAL_CORRECTION=0.03f; // Turn and RPM feedback COMBINED stay within 3%.
const int MAX_CORRECTION_PWM=5; // Also preserve the user's less-than-10 PWM difference.
const bool ENABLE_STARTUP_KICK=true;
const uint32_t KICK_TIME=150, COMMAND_TIMEOUT=500, REVERSE_DEADTIME=120;
const uint32_t CONTROL_INTERVAL=20, RPM_INTERVAL=200;
const float PWM_RISE_PER_SECOND=100.0f, PWM_FALL_PER_SECOND=240.0f;

// Feedback trims RPM ratio around calibrated feedforward PWM. No guessed absolute RPM target.
const bool ENABLE_RPM_SYNC=true;
const float SYNC_KP=0.015f, SYNC_KI=0.005f, SYNC_DEADZONE=2.0f, MAX_SYNC_TRIM=0.5f;
volatile uint32_t leftCount=0, rightCount=0;
uint32_t lastLeftCount=0, lastRightCount=0;
float rpmLeft=0, rpmRight=0, syncIntegral=0, syncTrim=0;
float currentSpeed=0, currentTurn=0, actualLeftPWM=0, actualRightPWM=0;
float targetLeftPWM=0, targetRightPWM=0;
int direction=0, appliedDirection=0, manualSteering=0;
bool manualMode=false, reversing=false, kickActive=false, feedbackWindow=false;
bool sessionReady=false;
uint32_t lastCommand=0, lastControl=0, lastRPM=0, reverseStarted=0, kickStarted=0;
char buffer[96]; size_t length=0; bool discard=false;

void IRAM_ATTR leftISR(){ ++leftCount; }
void IRAM_ATTR rightISR(){ ++rightCount; }
float bound(float v,float lo,float hi){ return fminf(hi,fmaxf(lo,v)); }
float correctionLower(float base){
  const float nominal=(float)lroundf(base);
  const float limit=fminf(nominal*MAX_TOTAL_CORRECTION,(float)MAX_CORRECTION_PWM);
  return ceilf(nominal-limit);
}
float correctionUpper(float base){
  const float nominal=(float)lroundf(base);
  const float limit=fminf(nominal*MAX_TOTAL_CORRECTION,(float)MAX_CORRECTION_PWM);
  return floorf(nominal+limit);
}

void setDirection(int value){
  digitalWrite(IN1,value>0?HIGH:LOW); digitalWrite(IN2,value<0?HIGH:LOW);
  digitalWrite(IN3,value>0?HIGH:LOW); digitalWrite(IN4,value<0?HIGH:LOW);
  appliedDirection=value;
}
void writePWM(float left,float right){
  actualLeftPWM=bound(left,0,MAX_PWM); actualRightPWM=bound(right,0,MAX_PWM);
  analogWrite(ENA,(int)lroundf(actualLeftPWM)); analogWrite(ENB,(int)lroundf(actualRightPWM));
}
void stopMotors(){
  currentSpeed=currentTurn=0; direction=manualSteering=0;
  targetLeftPWM=targetRightPWM=0;
  syncIntegral=syncTrim=0; kickActive=reversing=feedbackWindow=false;
  writePWM(0,0); setDirection(0);
}
void basePWM(float speed,float &left,float &right){
  float f=bound((speed-START_SPEED)/(CALIBRATED_SPEED-START_SPEED),0,1);
  left=LEFT_MIN_PWM+f*(LEFT_BASE_PWM-LEFT_MIN_PWM);
  right=RIGHT_MIN_PWM+f*(RIGHT_BASE_PWM-RIGHT_MIN_PWM);
}
void wheelFactors(float &left,float &right){
  left=right=1;
  if(manualMode && direction>0 && abs(manualSteering)>=30){
    if(manualSteering>0) left=0; else right=0;
  } else {
    float reduction=MAX_TURN_REDUCTION*fabsf(currentTurn);
    if(currentTurn>0) left=1-reduction;
    if(currentTurn<0) right=1-reduction;
  }
}
float slew(float oldValue,float target,float dt){
  float limit=(target>oldValue?PWM_RISE_PER_SECOND:PWM_FALL_PER_SECOND)*dt;
  return oldValue+bound(target-oldValue,-limit,limit);
}
void drive(float speed,float turn,int requestedDirection,bool manual,int steering){
  if(speed<=0){ stopMotors(); return; }
  bool starting=direction==0;
  if(appliedDirection!=0 && requestedDirection!=appliedDirection){
    writePWM(0,0); setDirection(0); reversing=true; reverseStarted=millis();
    kickActive=false; syncIntegral=syncTrim=0; feedbackWindow=false;
  }
  if(fabsf(turn-currentTurn)>0.15f || manual!=manualMode) feedbackWindow=false;
  currentSpeed=speed; currentTurn=turn; direction=requestedDirection;
  manualMode=manual; manualSteering=steering;
  if(starting && !reversing && ENABLE_STARTUP_KICK){ kickActive=true; kickStarted=millis(); }
}
void updateMotors(){
  uint32_t now=millis();
  if(direction && (uint32_t)(now-lastCommand)>=COMMAND_TIMEOUT){
    stopMotors(); Serial.println("EMERGENCY COMMAND TIMEOUT");
  }
  if((uint32_t)(now-lastControl)<CONTROL_INTERVAL) return;
  float dt=fminf((uint32_t)(now-lastControl)/1000.0f,0.05f); lastControl=now;
  if(!direction) return;
  if(reversing){
    if((uint32_t)(now-reverseStarted)<REVERSE_DEADTIME) return;
    reversing=false;
    if(ENABLE_STARTUP_KICK){ kickActive=true; kickStarted=now; }
  }
  setDirection(direction);
  float left,right,lf,rf; basePWM(currentSpeed,left,right); wheelFactors(lf,rf);
  const float baseLeft=left, baseRight=right;
  left*=lf; right*=rf;
  // Positive trim slows left / speeds right to achieve the requested RPM ratio.
  if(ENABLE_RPM_SYNC && direction==1 && lf>0 && rf>0){ left-=syncTrim; right+=syncTrim; }
  // Manual pivot is deliberately one wheel stopped; all moving-wheel corrections
  // otherwise stay within +/-3% of that wheel's straight-driving baseline.
  if(lf>0) left=bound(left,correctionLower(baseLeft),correctionUpper(baseLeft));
  if(rf>0) right=bound(right,correctionLower(baseRight),correctionUpper(baseRight));
  left=bound(left,0,MAX_PWM); right=bound(right,0,MAX_PWM);
  targetLeftPWM=left; targetRightPWM=right;
  if(kickActive && (uint32_t)(now-kickStarted)<KICK_TIME){
    feedbackWindow=false;
    writePWM(lf>0?fmaxf(left,LEFT_KICK_PWM):0,rf>0?fmaxf(right,RIGHT_KICK_PWM):0);
    return;
  }
  kickActive=false;
  float nextLeft=slew(actualLeftPWM,left,dt), nextRight=slew(actualRightPWM,right,dt);
  if(direction!=1 || lf==0 || rf==0 || fabsf(nextLeft-left)>1 || fabsf(nextRight-right)>1)
    feedbackWindow=false;
  writePWM(nextLeft,nextRight);
}
void updateRPM(){
  uint32_t now=millis();
  if((uint32_t)(now-lastRPM)<RPM_INTERVAL) return;
  float dt=(uint32_t)(now-lastRPM)/1000.0f; lastRPM=now;
  uint32_t left=leftCount,right=rightCount,dl=left-lastLeftCount,dr=right-lastRightCount;
  lastLeftCount=left; lastRightCount=right;
  float rawLeft=dl*60.0f/(LEFT_COUNTS_PER_REV*dt), rawRight=dr*60.0f/(RIGHT_COUNTS_PER_REV*dt);
  rpmLeft=0.5f*rpmLeft+0.5f*rawLeft; rpmRight=0.5f*rpmRight+0.5f*rawRight;
  float lf,rf; wheelFactors(lf,rf);
  if(ENABLE_RPM_SYNC && feedbackWindow && direction==1 && lf>0 && rf>0 && dl>0 && dr>0){
    float error=rpmLeft/lf-rpmRight/rf;
    if(fabsf(error)<SYNC_DEADZONE) error=0;
    // Integral bound prevents windup; freeze integration if output pushes a PWM limit.
    float bl,br; basePWM(currentSpeed,bl,br);
    // Freeze integration when it would push farther into the total correction cap.
    bool saturated=error>0
      ? (targetLeftPWM<=correctionLower(bl)+0.1f || targetRightPWM>=correctionUpper(br)-0.1f)
      : (targetLeftPWM>=correctionUpper(bl)-0.1f || targetRightPWM<=correctionLower(br)+0.1f);
    if(!saturated) syncIntegral=bound(syncIntegral+SYNC_KI*error*dt,-MAX_SYNC_TRIM,MAX_SYNC_TRIM);
    syncTrim=bound(SYNC_KP*error+syncIntegral,-MAX_SYNC_TRIM,MAX_SYNC_TRIM);
  }
  feedbackWindow=direction==1 && !kickActive && !reversing && lf>0 && rf>0
                 && fabsf(actualLeftPWM-targetLeftPWM)<1 && fabsf(actualRightPWM-targetRightPWM)<1;
  Serial.print("STATUS SPD="); Serial.print(currentSpeed,3);
  Serial.print(" TURN="); Serial.print(currentTurn,4);
  Serial.print(" PWM_L="); Serial.print(actualLeftPWM,1);
  Serial.print(" PWM_R="); Serial.print(actualRightPWM,1);
  Serial.print(" RPM_L="); Serial.print(rpmLeft,2);
  Serial.print(" RPM_R="); Serial.print(rpmRight,2);
  Serial.print(" TRIM="); Serial.println(syncTrim,2);
}
bool number(const char *s,float &v){ char *end; v=strtof(s,&end); return end!=s && *end=='\0' && isfinite(v); }
void process(char *line){
  if(strcmp(line,"STOP")==0 || strcmp(line,"CTRL,0.000,999")==0){ stopMotors(); lastCommand=millis(); return; }
  if(strcmp(line,"HELLO")==0){ stopMotors(); sessionReady=true; Serial.println("READY CENTERING V10"); return; }
  if(!sessionReady) return; // Never resume a pre-reboot drive before handshake.
  char *field[5]; int count=1; field[0]=line;
  for(char *p=line;*p;++p) if(*p==','){
    if(count>=5) return; *p='\0'; field[count++]=p+1;
  }
  bool manual=strcmp(field[0],"MANUAL")==0;
  if((manual && count!=4) || (!manual && (strcmp(field[0],"DRIVE")!=0 || count!=3))) return;
  float speed,turn; int requestedDirection=1,steering=0;
  if(!number(field[manual?2:1],speed) || speed<0 || speed>CALIBRATED_SPEED) return;
  if(manual){
    if(strcmp(field[1],"F")==0) requestedDirection=1;
    else if(strcmp(field[1],"B")==0) requestedDirection=-1; else return;
    float parsed;
    if(!number(field[3],parsed) || parsed!=floorf(parsed) || fabsf(parsed)>45) return;
    steering=(int)parsed; turn=0;
  } else if(!number(field[2],turn) || fabsf(turn)>1) return;
  lastCommand=millis(); drive(speed,turn,requestedDirection,manual,steering);
}
void readSerial(){
  int budget=64;
  while(budget-- && Serial.available()){
    char c=(char)Serial.read();
    if(c=='\n'){
      if(!discard){ buffer[length]='\0'; process(buffer); }
      discard=false; length=0;
    } else if(c!='\r' && !discard){
      if(c=='\0' || length>=sizeof(buffer)-1){ discard=true; length=0; }
      else buffer[length++]=c;
    }
  }
}
void setup(){
  Serial.begin(115200);
  for(int pin:{ENA,IN1,IN2,ENB,IN3,IN4}) pinMode(pin,OUTPUT);
  for(int pin:{LEFT_ENC_A,LEFT_ENC_B,RIGHT_ENC_A,RIGHT_ENC_B}) pinMode(pin,INPUT_PULLUP);
  stopMotors();
  attachInterrupt(digitalPinToInterrupt(LEFT_ENC_A),leftISR,RISING);
  attachInterrupt(digitalPinToInterrupt(RIGHT_ENC_A),rightISR,RISING);
  lastCommand=lastControl=lastRPM=millis();
  Serial.println("BOOT CENTERING V10");
}
void loop(){ readSerial(); updateMotors(); updateRPM(); delay(1); }
