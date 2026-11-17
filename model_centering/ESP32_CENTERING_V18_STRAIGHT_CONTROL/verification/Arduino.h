#pragma once
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <algorithm>
using std::min; using std::max; using std::isfinite;
#define IRAM_ATTR
#define HIGH 1
#define LOW 0
#define OUTPUT 1
#define INPUT_PULLUP 2
#define RISING 1
#define constrain(x,a,b) ((x)<(a)?(a):((x)>(b)?(b):(x)))
inline unsigned long millis(){return 0;}
inline void delay(int){}
inline void digitalWrite(int,int){}
inline void analogWrite(int,int){}
inline void pinMode(int,int){}
inline int digitalPinToInterrupt(int n){return n;}
inline void attachInterrupt(int,void(*)(),int){}
inline void analogWriteResolution(int,int){}
inline void analogWriteFrequency(int,int){}
inline void noInterrupts(){}
inline void interrupts(){}
struct MockSerial {void begin(int){} int available(){return 0;} int read(){return 0;} template<class... T> void print(T...){} template<class... T> void println(T...){} };
static MockSerial Serial;
