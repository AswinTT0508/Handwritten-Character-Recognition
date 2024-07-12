# Handwritten-Character-Recognition
This repository contains a simple implementation of a handwritten digit recognition system using a Convolutional Neural Network (CNN) with the MNIST dataset. The application allows users to draw digits on a canvas and recognizes the digit using a trained model.

**Table of Contents**:
* Introduction
* Installation
* Usage
* Model Training
* License
  
**Introduction**
This project demonstrates a basic CNN for recognizing handwritten digits. The model is trained on the MNIST dataset and can predict digits drawn on a Tkinter canvas. The project consists of two main parts:

1. Training the CNN model.
2. Creating a GUI application to draw and recognize digits.
   
**Installation**
**Prerequisites**
* Python 3.x
*Required Python libraries:
* Keras
* numpy
* pillow
* tkinter
* win32gui (part of pywin32)

**Install Required Libraries:**
pip install keras numpy pillow tk pywin32

**Clone the Repository:**
git clone https://github.com/AswinTT0508/handwritten-character-recognition.git
cd handwritten-character-recognition

**Usage**
**Running the Application**

To run the digit recognition application, execute the following command:
python app.py

A window will appear with a canvas where you can draw digits. Click the "Recognise" button to recognize the drawn digit, and the predicted digit along with the confidence score will be displayed.

**Model Training**
To train the model, use the following script:

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Define model parameters
batch_size = 128
num_classes = 10
epochs = 10

# Build the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model
model.save('mnist2.h5')
print("Saving the model as mnist2.h5")
