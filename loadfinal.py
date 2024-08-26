import numpy as np
import cv2
from keras.models import load_model
import serial
import time
import serial.tools.list_ports

frameWidth = 640         # Rezolutia camerei
frameHeight = 480       
brightness = 180         # Luminozitatea camerei
threshold = 0.75         # Pragul minim al probabilitatii
font = cv2.FONT_HERSHEY_SIMPLEX

# setarea camerei USB
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# afisare porturi disponibile
def list_serial_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

print("Available ports:", list_serial_ports())


# setarea comunicatiei seriale cu Arduino
try:
    arduino = serial.Serial('COM9', 9600)  # COM9 setat ca port
    time.sleep(2)  # asteptare initializare comunicatie seriala
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit()


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

# import model
model = load_model("model_trained.h5")

def getClassName(classNo):
    class_names = ['Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h', 
                   'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 
                   'Speed Limit 120 km/h', 'No passing', 'No passing for vehicles over 3.5 metric tons', 
                   'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles', 
                   'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left', 
                   'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 
                   'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 
                   'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead', 
                   'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 
                   'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons']
    return class_names[classNo]

def send_command(command):
    arduino.write(command.encode())
    print(f"Sent command: {command}")

while True:
    # preluare imagine de la camera
    success, imgOriginal = cap.read()
    
    # proecesarea imaginii
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)
    cv2.putText(imgOriginal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOriginal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    # predictia imaginii
    predictions = model.predict(img)
    classIndex = np.argmax(predictions)
    probabilityValue = np.amax(predictions)
    if probabilityValue > threshold:
        className = getClassName(classIndex)
        cv2.putText(imgOriginal, str(classIndex) + " " + className, (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        if classIndex == 14:  # STOP
            send_command('S')
        elif classIndex == 35:  # Ahead only
            send_command('x')
        elif classIndex == 5:  # Speed Limit 80 km/h
            send_command('8')
        elif classIndex == 1:  # Speed Limit 30 km/h
            send_command('3')
        elif classIndex == 33:  # Turn right ahead
            send_command('y')
        elif classIndex == 34:  # Turn left ahead
            send_command('z')
    else:
        send_command('N')  # Niciun semn recunoscut => control manual
    
    cv2.imshow("Result", imgOriginal)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()
