# Smart Vehicles: Using Arduino for Remote Control

## 📋 Project Overview
Welcome to my **Diploma Project** titled **Smart Vehicles: Using Arduino for Remote Control**. This project involves designing and building a smart car using **Arduino Mega 2560** as the core controller. The car integrates multiple sensors, machine learning, and components, enabling features such as remote control, traffic sign detection, real-time temperature measurement, automatic headlights, and pothole detection.

---

## ⚙️ Features
### 1. **Arduino Mega 2560**
- Selected for its **large number of analog and digital pins**, allowing for seamless integration of multiple sensors and components.

### 2. **H-Bridge Circuit**
- Drives the **DC motors** responsible for wheel rotation and directional control.

### 3. **Photoresistor Sensor**
- Automatically turns on the **LED lights** when low light conditions are detected.

### 4. **Ultrasonic Sensor**
- Detects **potholes** or obstacles ahead of the car, providing safety features.

### 5. **Temperature Sensor**
- Measures the **engine temperature** in real time, ensuring safe operation and performance monitoring.

### 6. **Bluetooth and Wired Control**
- Allows for both **wireless communication** (via Bluetooth) and **wired control**.

### 7. **Traffic Sign Detection with External Webcam**
- An **external webcam** detects traffic signs using a custom-trained **machine learning model**.
- The car's behavior changes dynamically based on the detected traffic sign (e.g., stop, turn left/right, or reduce speed).

### 8. **Machine Learning Model**
- A custom-trained **traffic sign recognition model** enables real-time decision-making.

### 9. **LED Bulbs**
- Four **LED bulbs** are used as car headlights or indicators, which can be automated via sensors.

### 10. **Batteries**
- Two **3.7V batteries** for power supply.
- One **9.6V battery** for additional power requirements.

### 11. **Additional Components**
- Other components such as resistors, wires, and switches, as shown in the circuit diagram.

---

## 🔌 Circuit Diagram
The following diagram illustrates the full hardware setup:

![Circuit Diagram](./schematz.png)

---

## 🛠️ Components Used
- **Arduino Mega 2560**
- **L298N H-Bridge Motor Driver**
- **Photoresistor Sensor**
- **Ultrasonic Sensor (HC-SR04)**
- **Temperature Sensor (DHT11 or similar)**
- **Bluetooth Module (HC-05)**
- **External Webcam**
- **DC Motors**
- **4 LED Bulbs**
- **3.7V Batteries (x2)**
- **9.6V Battery**
- **Custom Traffic Sign Recognition Model**
- **Resistors, Jumper Wires, and Breadboard**

---

## 🧰 Setup Instructions
1. **Hardware Assembly**:
   - Connect all components as shown in the circuit diagram.
   - Ensure proper power connections to the Arduino and motor driver.

2. **Software Requirements**:
   - Install **Arduino IDE**.
   - Install required libraries for the sensors, Bluetooth module, and webcam integration.
   - Install **Python** and the necessary machine learning libraries (e.g., TensorFlow, OpenCV).

3. **Code Upload**:
   - Upload the provided code to the **Arduino Mega 2560** using Arduino IDE.
   - Run the **traffic sign detection** Python script on your external system connected to the webcam.

4. **Power Up**:
   - Insert the batteries (3.7V and 9.6V) into the circuit.
   - Test all connections and ensure proper functioning.

5. **Control Modes**:
   - **Bluetooth**: Pair the HC-05 Bluetooth module with a mobile phone or computer for wireless control.
   - **Wired Control**: Connect the external webcam to detect traffic signs and adjust car behavior.

6. **Traffic Sign Detection**:
   - Use the provided **machine learning model** to identify traffic signs.
   - Adjust the car's movement dynamically based on the detected sign (e.g., stop, turn, or slow down).

---

## 🚀 How It Works
1. **Motor Control**:
   - The H-Bridge circuit controls the movement of the car's motors, enabling forward, backward, left, and right movements.

2. **Automatic Headlights**:
   - The photoresistor sensor detects light levels and activates the LED lights in low light conditions.

3. **Pothole Detection**:
   - The ultrasonic sensor detects nearby potholes or obstacles and can trigger alerts.

4. **Temperature Monitoring**:
   - The temperature sensor measures real-time engine temperature and can display data via a connected LCD.

5. **Bluetooth and Wired Communication**:
   - Bluetooth enables wireless control, while the wired connection allows webcam integration for advanced decision-making.

6. **Traffic Sign Detection**:
   - The external webcam captures live video.
   - The **machine learning model** processes the video stream to identify traffic signs.
   - Based on the detected sign, the car adjusts its behavior (e.g., stopping at a stop sign or turning at a turn sign).

---

## 📂 Repository Structure
```
├── arduino_for_car/       # Arduino sketch files
├── myData/                # Training data for traffic sign detection
├── ModelTraining.ipynb    # Jupyter Notebook for training the model
├── model.h5               # Final trained machine learning model
├── model_reconstructed.h5 # Reconstructed model for deployment
├── labels.csv             # Labels for traffic sign classes
├── loadfinal.py           # Script to load and test the model
├── traffictest.py         # Real-time traffic sign detection script
├── schematz.jpg           # Hardware circuit diagram
├── README.md              # Project documentation
└── LICENSE                # License file (if applicable)
```

---

## 📸 Demo & Testing
- Add photos or videos showcasing the project in action.
- Demonstrate features such as **Bluetooth control**, **automatic headlights**, **pothole detection**, and **traffic sign recognition**.

---

## 🔗 Future Improvements
- Integrate a GPS module for location tracking.
- Add a **mobile app** interface for better user control.
- Improve traffic sign detection accuracy with advanced models.
- Use onboard cameras for fully autonomous navigation.

---


## 🤝 Acknowledgments
- **Arduino Community** for documentation and libraries.
- **OpenCV** and **TensorFlow** for enabling traffic sign recognition.
- My mentors, professors, and peers for guidance and feedback.

---

## 📨 Contact
If you have any questions or suggestions, feel free to contact me:
- **Email**: dan.oprean.77@gmail.com
- **GitHub**: [GitHubProfile](https://github.com/thedanoprean)

---

Thank you for visiting my project repository! 🚗✨
