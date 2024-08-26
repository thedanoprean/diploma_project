#include <SoftwareSerial.h>
#include <LiquidCrystal.h>
#include <dht.h>

// Variables for DHT
#define DHTPin 8
dht DHT;

int ledPin = A15;   // the number of the LED pin
int ldrPin = A14;   // the number of the LDR pin
// rgb pins
int rgb1 = 48;
int rgb2 = 50;
int rgb3 = 52;
// Front Motor Pins  
int Enable1 = 7;
int Motor1_Pin1 = 6;  
int Motor1_Pin2 = 5;  
// Back Motor Pins      
int Motor2_Pin1 = 4; 
int Motor2_Pin2 = 3;
int Enable2 = 2;
// ultrasonic
int trigPin = 44;
int echoPin = 46;
// bluetooth
int bluetoothTx = 0; 
int bluetoothRx = 1; 

const int rs = 22, en = 23, d4 = 24, d5 = 25, d6 = 26, d7 = 28;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

int velocity;

void setup() {
  Serial.begin(9600);
  pinMode(ledPin, OUTPUT);  
  pinMode(ldrPin, INPUT);

  pinMode(rgb1, OUTPUT);
  pinMode(rgb2, OUTPUT);
  pinMode(rgb3, OUTPUT);

  pinMode(Enable1, OUTPUT);
  pinMode(Motor1_Pin1, OUTPUT);  
  pinMode(Motor1_Pin2, OUTPUT);  
  pinMode(Motor2_Pin1, OUTPUT);  
  pinMode(Motor2_Pin2, OUTPUT);
  pinMode(Enable2, OUTPUT);

  analogWrite(Enable1, 255);
  analogWrite(Enable2, 0);
  digitalWrite(Motor2_Pin2, LOW);
  digitalWrite(Motor2_Pin1, LOW);
  digitalWrite(Motor1_Pin2, LOW);
  digitalWrite(Motor1_Pin1, LOW);

  pinMode(trigPin, OUTPUT); 
  pinMode(echoPin, INPUT); 

  // set up the LCD
  lcd.begin(16, 2);
  analogWrite(9, 100);

  velocity = 255;
}

void loop() {
  char c;
  if (Serial.available() > 0) {
    c = Serial.read();
    Serial.println(c);
    switch(c) {
      case 'F':  // Moving the Car Forward Bluetooth
        UltraSonic();
        analogWrite(Enable2, velocity);
        digitalWrite(Motor2_Pin2, LOW);
        digitalWrite(Motor2_Pin1, HIGH);
        digitalWrite(Motor1_Pin1, LOW);
        digitalWrite(Motor1_Pin2, LOW);   
        break;
      case 'x':  // Moving the Car Forward Python
        UltraSonic();
        analogWrite(Enable2, velocity);
        digitalWrite(Motor2_Pin2, LOW);
        digitalWrite(Motor2_Pin1, HIGH);
        digitalWrite(Motor1_Pin1, LOW);
        digitalWrite(Motor1_Pin2, LOW);   
        break;
      case 'B':  // Moving the Car Backward Bluetooth
        analogWrite(Enable2, velocity);
        digitalWrite(Motor2_Pin1, LOW);
        digitalWrite(Motor2_Pin2, HIGH);
        digitalWrite(Motor1_Pin1, LOW);
        digitalWrite(Motor1_Pin2, LOW);
        UltraSonic();
        break;
      case 'L':  // Steer Wheels Right Bluetooth
        digitalWrite(Motor1_Pin1, LOW);
        digitalWrite(Motor1_Pin2, HIGH);
        digitalWrite(Motor2_Pin1, LOW);
        digitalWrite(Motor2_Pin2, LOW);
        UltraSonic();
        break;
      case 'z':  // Moving the Car Front-Left Python
        analogWrite(Enable2, velocity);
        digitalWrite(Motor1_Pin1, LOW);
        digitalWrite(Motor1_Pin2, HIGH);
        digitalWrite(Motor2_Pin1, HIGH);
        digitalWrite(Motor2_Pin2, LOW);
        UltraSonic();
        break;
      case 'R':  // Steer Wheels Right Bluetooth
        digitalWrite(Motor1_Pin2, LOW);
        digitalWrite(Motor1_Pin1, HIGH);  
        digitalWrite(Motor2_Pin1, LOW);
        digitalWrite(Motor2_Pin2, LOW);
        UltraSonic();
        break;
      case 'y':  // Moving the Car Front-Right Python
        analogWrite(Enable2, velocity);
        digitalWrite(Motor1_Pin2, LOW);
        digitalWrite(Motor1_Pin1, HIGH);  
        digitalWrite(Motor2_Pin1, HIGH);
        digitalWrite(Motor2_Pin2, LOW);
        UltraSonic();
        break;
      case 'S':  // Stop
        digitalWrite(Motor2_Pin2, LOW);
        digitalWrite(Motor2_Pin1, LOW);
        digitalWrite(Motor1_Pin2, LOW);
        digitalWrite(Motor1_Pin1, LOW);
        break; 
      case 'I':  // Moving the Car Forward Right Bluetooth
        analogWrite(Enable2, velocity);
        digitalWrite(Motor2_Pin2, LOW);
        digitalWrite(Motor2_Pin1, HIGH);
        digitalWrite(Motor1_Pin2, LOW);
        digitalWrite(Motor1_Pin1, HIGH);
        UltraSonic();
        break; 
      case 'J':  // Moving the Car Backwards Right
        analogWrite(Enable2, velocity);
        digitalWrite(Motor1_Pin2, LOW);
        digitalWrite(Motor1_Pin1, HIGH);
        digitalWrite(Motor2_Pin1, LOW);
        digitalWrite(Motor2_Pin2, HIGH);
        UltraSonic();
        break;        
      case 'G':  // Moving the Car Forward Left
        analogWrite(Enable2, velocity);
        digitalWrite(Motor2_Pin2, LOW);
        digitalWrite(Motor2_Pin1, HIGH);
        digitalWrite(Motor1_Pin1, LOW);
        digitalWrite(Motor1_Pin2, HIGH);
        UltraSonic();
        break; 
      case 'H':  // Moving the Car Backwards Left
        analogWrite(Enable2, velocity);
        digitalWrite(Motor2_Pin1, LOW);
        digitalWrite(Motor2_Pin2, HIGH);
        digitalWrite(Motor1_Pin1, LOW);
        digitalWrite(Motor1_Pin2, HIGH);
        UltraSonic();
        break;
      case '8': // Setting speed 80 km/h
        velocity = (c - '0') * 25;
        analogWrite(Enable2, velocity);
        digitalWrite(Motor2_Pin2, LOW);
        digitalWrite(Motor2_Pin1, HIGH);
        digitalWrite(Motor1_Pin1, LOW);
        digitalWrite(Motor1_Pin2, LOW);
      case '3': // Setting speed 30 km/h
        velocity = (c - '0') * 25;
        analogWrite(Enable2, velocity);
        digitalWrite(Motor2_Pin2, LOW);
        digitalWrite(Motor2_Pin1, HIGH);
        digitalWrite(Motor1_Pin1, LOW);
        digitalWrite(Motor1_Pin2, LOW);
      case 'W':  // Front light ON 
        digitalWrite(ledPin, HIGH);
        break;
      case 'w':  // Front light OFF
        digitalWrite(ledPin, LOW);
        break;
      case 'U':  // Back light ON 
        LCD();
        break;
      case 'u':  // Back light OFF 
        lcd.clear();
        lcd.display();
        break; 
      case 'N':  // No recognized sign, allow manual control
        break; 
      // Controlling the Speed of Car  
      default:  // Get velocity
        if (c == 'q') {
          velocity = 255;  // max speed
        } else if (c >= '0' && c <= '9') {
          velocity = (c - '0') * 25;
        analogWrite(Enable2, velocity);
        }
    }
  }
  LDR();
}

void LDR() {  
  int ldrStatus = analogRead(ldrPin);   
  if (ldrStatus >= 300) {
    digitalWrite(ledPin, HIGH);  
    digitalWrite(rgb1, HIGH);
    digitalWrite(rgb2, HIGH);
    digitalWrite(rgb3, HIGH); // turn LED on
  } else {
    digitalWrite(ledPin, LOW);  
    digitalWrite(rgb1, LOW);
    digitalWrite(rgb2, LOW);
    digitalWrite(rgb3, LOW); // turn LED off
  }
}

void LCD() {
  // get data from DHT
  int Data = DHT.read11(DHTPin);

  // variables for temp in C and F degrees and humidity
  double t = DHT.temperature;

  // printing the temp
  lcd.setCursor(0, 0);
  lcd.print("Temp.: ");
  lcd.print(t, 2);
  lcd.print((char)223);
  lcd.print("C");
  delay(1000);
}

void UltraSonic() {
  long duration;
  int distance;

  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH);
  distance = duration * 0.034 / 2;

  while(distance > 3) {
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    duration = pulseIn(echoPin, HIGH);
    distance = duration * 0.034 / 2;

    analogWrite(Enable1, 0);
    analogWrite(Enable2, 0);

    if (distance < 3) {
      analogWrite(Enable1, 255);
      analogWrite(Enable2, velocity);
      break;
    }

    Serial.print("Distance: ");
    Serial.print(distance);
    Serial.print(" cm. Speed value: ");
    Serial.println(velocity);
    
    delay(10);
  }
}
