import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


def getCNNModel(step_size):
    # model = models.Sequential()
    # model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(28,28,1)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.Dense(10, activation='softmax'))

    # model.compile(optimizer='adam',
    #             loss=tf.keras.losses.categorical_crossentropy,
    #             metrics=['accuracy'])
    # return model

    model = models.Sequential()
    #convolutional layer with rectified linear unit activation
    model.add(layers.Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(28,28,1)))
    #32 convolution filters used each of size 3x3
    #again
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #64 convolution filters used each of size 3x3
    #choose the best features via pooling
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    #randomly turn neurons on and off to improve convergence
    model.add(layers.Dropout(0.25))
    #flatten since too many dimensions, we only want a classification output
    model.add(layers.Flatten())
    #fully connected to get all relevant data
    model.add(layers.Dense(128, activation='relu'))
    #one more dropout for convergence' sake :) 
    model.add(layers.Dropout(0.5))
    #output a softmax to squash the matrix into output probabilities
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy'])
    return model