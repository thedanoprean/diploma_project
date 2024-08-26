import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import random

path = "myData"  # folder principal
labelFile = 'labels.csv'  # fisier cu numele claselor si numarul lor
batch_size_val = 50  # cate sa se proceseze o data
steps_per_epoch_val = 2000
epochs_val = 10
imageDimensions = (32, 32, 3)
testRatio = 0.2  # 20% test
validationRatio = 0.2  # 20% din cele 80% ramase pt validare


count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:", len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")
for x in range(0, len(myList)):
    myPicList = os.listdir(path + "/" + str(count))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)
        images.append(curImg)
        classNo.append(count)
    print(count, end=" ")
    count += 1
print(" ")
images = np.array(images)
classNo = np.array(classNo)

# Impartim datele
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

# x_train = sir al imaginilor de antrenat
# y_train = corespunde class Id

# verificare daca nr imaginilor se potriveste cu nr etichetelor pentru fiecare set de date
print("Data Shapes")
print("Train", end=""); print(X_train.shape, y_train.shape)
print("Validation", end=""); print(X_validation.shape, y_validation.shape)
print("Test", end=""); print(X_test.shape, y_test.shape)
assert(X_train.shape[0] == y_train.shape[0]), "The quantity of images does not match the quantity of labels in the training dataset."
assert(X_validation.shape[0] == y_validation.shape[0]), "The number of images does not match the number of labels in the validation set."
assert(X_test.shape[0] == y_test.shape[0]), "The number of images does not match the number of labels in the test set."
assert(X_train.shape[1:] == imageDimensions), "The dimensions of the training images are incorrect."
assert(X_validation.shape[1:] == imageDimensions), "The dimensions of the validation images are incorrect."
assert(X_test.shape[1:] == imageDimensions), "The dimensions of the test images are incorrect."

# citim fisierul CSV
data = pd.read_csv(labelFile)
print("data shape ", data.shape, type(data))

# preprocesarea imaginilor

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)  # convertire grayscale
    img = equalize(img)  # standardizare luminozitate in imagini
    img = img / 255  # normalizarea valorilor intre 0 si 1 in loc de 0 si 255
    return img

X_train = np.array(list(map(preprocessing, X_train)))  # iterare si preprocesare imagini
X_validation = np.array(list(map(preprocessing, X_validation)))
X_test = np.array(list(map(preprocessing, X_test)))

# adaugare adancime de 1
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# augumentarea imaginilor pentru a le face mai generice
dataGen = ImageDataGenerator(width_shift_range=0.1,  # 0.1 = 10% daca > 1 
                            height_shift_range=0.1,
                            zoom_range=0.2,  # 0.2 inseamna 0.8 - 1.2
                            shear_range=0.1,  # magnitudinea shear angle
                            rotation_range=10)  # degrees
dataGen.fit(X_train)

y_train = to_categorical(y_train, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# CNN Model
def myModel():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)  # kernelul care ia caractersiticile imaginilor 
                             # sterge 2 pixeli din fiecare border pt o imagine 32x32
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)  # pentru reducere overfitting
    no_Of_Nodes = 500  # numarul de noduri in layere ascunse
    model = Sequential()
    model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimensions[0], imageDimensions[1], 1), activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))  # nu afecteaza adancimea / nr de filtre

    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))  
    model.add(Dense(noOfClasses, activation='softmax'))  # output layer
    # compilare model
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# antrenare
model = myModel()
print(model.summary())
history = model.fit(dataGen.flow(X_train, y_train, batch_size=batch_size_val), steps_per_epoch=steps_per_epoch_val, epochs=epochs_val, validation_data=(X_validation, y_validation), shuffle=1)

score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# salvare model ca h5 
model.save("model_trained.h5")

# evaluare metrici

# Obtinere predictii model pe setul de test
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calculare matrice de confuzie
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Calculare metricile: accuracy, precision, recall (sau sensitivity), si f1-score
class_report = classification_report(y_true, y_pred_classes, output_dict=True)

# Extrage valorile din raportul de clasificare
accuracy = class_report['accuracy']
precision = class_report['macro avg']['precision']
recall = class_report['macro avg']['recall']
f1_score = class_report['macro avg']['f1-score']

# Extrage valorile de TP, FP, TN, FN pentru fiecare clasa
TP = np.diag(conf_matrix)
FP = conf_matrix.sum(axis=0) - TP
FN = conf_matrix.sum(axis=1) - TP
TN = conf_matrix.sum() - (FP + FN + TP)

# Construiste tabelul
evaluation_table = {
    'Class': list(range(noOfClasses)),
    'True Positive': TP,
    'False Positive': FP,
    'True Negative': TN,
    'False Negative': FN,
    'Precision': [class_report[str(i)]['precision'] for i in range(noOfClasses)],
    'Recall': [class_report[str(i)]['recall'] for i in range(noOfClasses)],
    'F1-Score': [class_report[str(i)]['f1-score'] for i in range(noOfClasses)],
    'Accuracy': [accuracy] * noOfClasses  # accuracy aceeasi pentru toate clasele
}

evaluation_df = pd.DataFrame(evaluation_table)

# Afiseaza tabelul folosind pandas
print(evaluation_df)

# Grafice pentru fiecare metrica
metrics = ['Precision', 'Recall', 'F1-Score', 'True Positive', 'False Positive', 'True Negative', 'False Negative']
num_metrics = len(metrics)

plt.figure(figsize=(15, 10))

for i, metric in enumerate(metrics, 1):
    plt.subplot(num_metrics // 2 + num_metrics % 2, 2, i)
    plt.bar(evaluation_df['Class'], evaluation_df[metric])
    plt.title(metric)
    plt.xlabel('Class')
    plt.ylabel(metric)

plt.tight_layout()
plt.show()

# Exporta fiecare metrica intr-un fisier Excel separat
for metric in metrics:
    evaluation_df[['Class', metric]].to_excel(f'{metric}.xlsx', index=False)
