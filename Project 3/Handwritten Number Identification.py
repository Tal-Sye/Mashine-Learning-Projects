#importing starting libraries and MNIST for dataset
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')

#create testing and training sets
from tensorflow.keras.datasets import mnist
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))
from tensorflow.keras.utils import to_categorical
trainY = to_categorical(trainY)
testY = to_categorical(testY)

#normalizing testing and training sets
normalized_train = trainX.astype('float32')
normalized_test = testX.astype('float32')
new_normalized_train = normalized_train / 255.0
new_normalized_test = normalized_test / 255.0
trainX = new_normalized_train
testX = new_normalized_test

#creating and adjusting the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))

#flatten the model
from tensorflow.keras.layers import Flatten
model.add(Flatten())
from tensorflow.keras.layers import Dense
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

#doing a gradient descent
from tensorflow.keras.optimizers import SGD
opt = SGD(learning_rate=0.01, momentum=0.9)

#compiling the model
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#running and building the model
#adjusting to reach highest accuracy as needed
#10 epochs being used

model_name = model.fit(new_normalized_train, trainY, validation_data=(new_normalized_test, testY), epochs=10, batch_size=64)
