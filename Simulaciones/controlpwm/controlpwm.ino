#include <PID_v1.h>
#include <TimerOne.h>  /
/*-------------------------------------Variables para Impresión en consola------------------------*/
byte         cmd       =  0;             // Use for serial comunication.  
byte         flags;                      // Flag for print values in the serial monitor
/*------------------------------Variebles for LM298N-------------------------------*/
int IN1  = 11; 
int IN2  = 10;
int PWM1 = 9;
int forWARDS  = 1; 
int backWARDS = 0;
float start   = 0;

/*------------------------------Variables for incremental encoder----------------------------------*/
volatile long contador =  0;   
byte          ant      =  0;    
byte          act      =  0;
const byte    encA     =  2;                  // Signal for channel A
const byte    encB     =  3;                  // Signal for channel B
int   MIN_MAX_POST     =  300;                 // Limit the maximun position
/*-----------------------------We defined variables for PID algorithm------------------------------*/
TimerOne PID_compute;                         // Timer
double Setpoint, Input, Output;
double SampleTime = 100;                      // time in mili seconds, RUN at 160MHZ ESP8266
//double Kp=5, Ki=2, Kd=0.001; 
//double Kp=5.5, Ki=3.6, Kd=0.002;
double Kp=5.5, Ki=3.6, Kd=0.002;              // PID gain
PID myPID(&Input, &Output, &Setpoint, Kp, Ki, Kd, DIRECT);

/*----------------------------- Interruption Function----------------------------------------------*/ 
void PID_DCmotor_interrup(){
  //Serial.print("t: ");Serial.println(millis()-start)   ;
  //start = millis();
  myPID.Compute();  // Calculus for PID algorithm 
  RunMotor(Output); // PWM order to DC driver
}

/*----------------------------SETUP----------------------------------------------------------------*/
void setup(){ 
  Serial.begin(9600); 
  //Iniciando L298N
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(PWM1, LOW);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(PWM1, OUTPUT);
  
  Setpoint = 100.0; // Init in position 0.0 
  //RUN the PID
  myPID.SetMode(AUTOMATIC);
  // Max Min values for PID algorithm
  myPID.SetOutputLimits(-1023,1023); 
  // Sample Time for PID
  myPID.SetSampleTime(SampleTime);     

  // Initializing Interruptions
  PID_compute.initialize(SampleTime);
  PID_compute.attachInterrupt(PID_DCmotor_interrup); 
  Serial.begin(9600);

  // Pin Interruption
  attachInterrupt(digitalPinToInterrupt(encA), encoder, CHANGE); // rising and falling flank
  attachInterrupt(digitalPinToInterrupt(encB), encoder, CHANGE); // rising and falling flank
}

/*----------------------------LOOP----------------------------------------------------------------*/
void loop(){  
  Serial.print("PWM  :");Serial.print(Output); 
  Serial.print(" |  contador  :");Serial.print(contador);
  Serial.print(" |  Setpoint  :");Serial.println(Setpoint);
  // Protection code
  //limit_post();
  // Ask for input data for change PID gain or setpoint
  //input_data();
}
/*------------------------------------------------------------------------------------------------*/

// Function for run the motor, backward, forward or stop
void RunMotor(double Usignal){  
  if (Setpoint-Input==0){
    shaftrev(IN1,IN2,PWM1,backWARDS, 0);
    //Serial.print("cero");
  }else if(Usignal>=0){
    shaftrev(IN1,IN2,PWM1,backWARDS, Usignal);
  }else{
      shaftrev(IN1,IN2,PWM1,forWARDS, -1*Usignal);
  }   
}

// Function that set DC_driver to move the motor
void shaftrev(int in1, int in2, int PWM, int sentido,int Wpulse){  
  if(sentido == 0){ //backWARDS
    digitalWrite(in2, HIGH);
    digitalWrite(in1, LOW);
    analogWrite(PWM,Wpulse);
    }
  if(sentido == 1){ //forWARDS
    digitalWrite(in2, LOW);
    digitalWrite(in1, HIGH);
    analogWrite(PWM,Wpulse);     
    }
}


void encoder(void){ 
  //Serial.println(ant);
  ant=act;                            // Saved act (current read) in ant (last read)
  act = digitalRead(encA)<<1|digitalRead(encB);
  if(ant==0 && act==1)        contador++;  // Increase the counter for forward movement
  else if(ant==1  && act==3)  contador++;
  else if(ant==3  && act==2)  contador++;
  else if(ant==2  && act==0)  contador++;
  else contador--;                         // Reduce the counter for backward movement

  // Enter the counter as input for PID algorith
  Input=contador;
}
