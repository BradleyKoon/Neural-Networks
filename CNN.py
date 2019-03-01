from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn import preprocessing 
from sklearn.decomposition import PCA 
from keras.utils import to_categorical 
from sklearn.model_selection import train_test_split
import loadData as ld  
import keras 

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

import matplotlib.pyplot as plt 
import numpy as np

import time 

# Wrapper for a convolutional neural net
# Conv2D layers start at conv_size and scale down linearly
# I.E, Size: 64, Layers: 2, Result: 64 Neuron layer -> 32 Neuron layer
class ConvNN:
    def originalCNN(self, shape, num_classes):
        #create model
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=shape))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(num_classes, activation='softmax'))

        #compile model using accuracy to measure model performance
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary() 

    def ChangeFilterCNN(self, shape, num_classes):
        #create model
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=shape))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(32, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(num_classes, activation='softmax'))

        #compile model using accuracy to measure model performance
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary() 

    def PaddedCNN(self, shape, num_classes):
        #create model
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=shape,padding="same"))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(num_classes, activation='softmax'))

        #compile model using accuracy to measure model performance
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary() 

    def DropoutCNN(self, shape, num_classes):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=shape))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.4))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        self.model.add(Dense(num_classes, activation='softmax'))

        #compile model using accuracy to measure model performance
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary() 

    def SigmoidCNN(self, shape, num_classes):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),activation='sigmoid',input_shape=shape))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), activation='sigmoid'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(128, (3, 3), activation='sigmoid'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.4))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='sigmoid'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.3))
        self.model.add(Dense(num_classes, activation='softmax'))

        #compile model using accuracy to measure model performance
        self.model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary() 

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size, verbose=True, earlystopping=False):
        start_time = time.time() 
        if earlystopping == True:
            callback = keras.callbacks.EarlyStopping(monitor='val_loss', verbose=1)
            history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), shuffle=True, epochs=epochs, batch_size=batch_size, callbacks=[callback])
        else:
            history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), shuffle=True, epochs=epochs, batch_size=batch_size)
        elapsed_time = time.time() - start_time
        print('Time to Train: ', elapsed_time)
        if verbose is True:
            print(history.history)
        return history 

    def test(self, X_test, y_test, history):
        test_eval = self.model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', test_eval[0])
        print('Test accuracy:', test_eval[1])

        accuracy = history.history['acc']
        val_accuracy = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(accuracy))
        plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

    def predict(self, predict_data, verbose=True):
        return self.model.predict(predict_data, verbose=verbose)

def run(RGB_BOOL=True, MIN_IMG_PER_LABEL=20, TRAIN_RATIO=0.75, epochs=10, batch_size=50, earlyStopping=False, conv_system="OriginalCNN"):

    print("Getting dataset...")

    num_classes = 0
    expected_classes = ld.getNumClasses(MIN_IMG_PER_LABEL, RGB_BOOL)

    # There is a small chance that sklearn's train/test split will not capture all labels between the train/test set
    # This loop will re-grab the datasets if not all of the datasets were correctly captured
    while True:
        X_train, X_test, y_train, y_test = ld.getLFWCropData(MIN_IMG_PER_LABEL, RGB_BOOL, TRAIN_RATIO)
        num_classes = len(np.unique(y_train))
        if num_classes != expected_classes:
            print("Bad train/test split, reloading dataset...")
        else:
            break

    print("Got datasets!")

    # Image Normalization 
    X_train = X_train.astype('float32') 
    X_test = X_test.astype('float32')
    X_train = X_train/255.0 
    X_test = X_test/255.0

    # One hot encoding
    labelEncoder = preprocessing.LabelEncoder() 
    y_train = labelEncoder.fit_transform(y_train) 
    y_train_categorical = to_categorical(y_train)
    y_test = labelEncoder.fit_transform(y_test)
    y_test_categorical = to_categorical(y_test)

    # Displaying images before training
    classes = labelEncoder.inverse_transform(y_test) 
    plt.figure(figsize=(15,15)) 
    for index, (image, label) in enumerate(zip(X_test[0:12], classes[0:12])):
        plt.subplot(4, 3, index + 1) 
        plt.tight_layout() 
        plt.imshow(image, cmap=plt.cm.gray)
        plt.title("Label: %s\n" % label, fontsize=13)
    plt.show() 

    if RGB_BOOL == False:
        X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

    conv = ConvNN()
    # These are some different convolutional systems 
    if conv_system == "OriginalCNN":
        print("Running OriginalCNN")
        conv.originalCNN((X_train.shape[1],X_train.shape[2],X_train.shape[3]), num_classes) 
    elif conv_system == "PaddedCNN":
        print("Running PaddedCNN") 
        conv.PaddedCNN((X_train.shape[1],X_train.shape[2],X_train.shape[3]), num_classes) 
    elif conv_system == "DropoutCNN":
        print("Running DropoutCNN") 
        conv.DropoutCNN((X_train.shape[1],X_train.shape[2],X_train.shape[3]), num_classes) 
    elif conv_system == "ChangeFilterCNN":
        print("Running ChangeFilterCNN") 
        conv.ChangeFilterCNN((X_train.shape[1],X_train.shape[2],X_train.shape[3]), num_classes)
    elif conv_system == "SigmoidCNN":
        print("Running SigmoidCNN") 
        conv.SigmoidCNN((X_train.shape[1],X_train.shape[2],X_train.shape[3]), num_classes)
    else:
        print("This convolutional system does not exist.  Some availible options are:")
        print("OriginalCNN")
        print("PaddedCNN") 
        print("DropoutCNN") 
        print("ChangeFilterCNN") 
        print("SigmoidCNN") 
        return 

    history = conv.train(X_train, y_train_categorical, X_test, y_test_categorical, epochs, batch_size, earlystopping=earlyStopping)
    pred = conv.predict(X_test) 

    if RGB_BOOL == False:
        X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2])
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2])

    # Displaying images after training 
    classes = np.argmax(pred, axis=1) 
    classes = labelEncoder.inverse_transform(classes) 
    plt.figure(figsize=(15,15)) 
    for index, (image, label) in enumerate(zip(X_test[0:12], classes[0:12])):
        plt.subplot(4, 3, index + 1) 
        plt.tight_layout() 
        plt.imshow(image, cmap=plt.cm.gray)
        plt.title("Predicted: %s\n" % label, fontsize=13)
    plt.show() 

    if RGB_BOOL == False:
        X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

    conv.test(X_test, y_test_categorical, history)

# With Grayscale Images 
print("----------------TRAINING WITH GRAYSCALE IMAGES----------------")
run(RGB_BOOL=False, conv_system="OriginalCNN") 
run(RGB_BOOL=False, conv_system="PaddedCNN") 
run(RGB_BOOL=False, conv_system="DropoutCNN")

# With RGB Images 
print("----------------TRAINING WITH RGB IMAGES----------------")
run(RGB_BOOL=True, conv_system="OriginalCNN") 
run(RGB_BOOL=True, conv_system="PaddedCNN") 
run(RGB_BOOL=True, conv_system="DropoutCNN")