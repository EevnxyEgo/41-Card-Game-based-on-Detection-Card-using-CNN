
import numpy as np
import h5py
import os


from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

fullPathToDataset = \
    'C:/Users/arsen/PCV/Big Project/CNN/DatasetH5PY'
    
folder = r"C:/Users/arsen/PCV/Big Project/CNN/Dataset/Training/Images"
classes = ['ace of spades', 'two of spades', 'three of spades', 'four of spades', 'five of spades',
           'six of spades', 'seven of spades', 'eight of spades', 'nine of spades', 'ten of spades',
           'jack of spades', 'queen of spades', 'king of spades', 'ace of clubs', 'two of clubs', 
           'three of clubs', 'four of clubs', 'five of clubs', 'six of clubs', 'seven of clubs', 
           'eight of clubs', 'nine of clubs', 'ten of clubs', 'jack of clubs', 'queen of clubs', 
           'king of clubs', 'ace of diamonds', 'two of diamonds', 'three of diamonds', 
           'four of diamonds', 'five of diamonds', 'six of diamonds', 'seven of diamonds', 
           'eight of diamonds', 'nine of diamonds','ten of diamonds', 'jack of diamonds', 
           'queen of diamonds', 'king of diamonds', 'ace of hearts', 'two of hearts', 
           'three of hearts', 'four of hearts', 'five of hearts', 'six of hearts', 'seven of hearts', 
           'eight of hearts', 'nine of hearts', 'ten of hearts', 'jack of hearts', 'queen of hearts', 
           'king of hearts']

length = len(classes)
print(length)

with h5py.File('C:/Users/arsen/PCV/Big Project/CNN/DatasetH5PY/datasetCardsGrayNormalized.hdf5', 'r') as f:
    
    print(list(f.keys()))
    
    #training
    x_train = f['x_train']  
    y_train = f['y_train']  

    x_train = np.array(x_train)  
    y_train = np.array(y_train) 
    
    #validation
    x_validation = f['x_validation']  
    y_validation = f['y_validation']  

    x_validation = np.array(x_validation)  
    y_validation = np.array(y_validation) 
    
    #test
    x_test = f['x_test']  
    y_test = f['y_test']  

    x_test = np.array(x_test)  
    y_test = np.array(y_test)  
    
y_train = to_categorical(y_train, num_classes = length)
y_validation = to_categorical(y_validation, num_classes = length)

print(y_train.shape)
print(y_validation.shape)

model = Sequential()

model.add(Conv2D(8, kernel_size=5, padding='same', activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPool2D())

model.add(Conv2D(16, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train,
                  batch_size=50,
                  epochs=5, 
                  validation_data=(x_validation, y_validation),
                  verbose=1)

os.chdir(fullPathToDataset)

model.save('ALLINONECoba.h5')