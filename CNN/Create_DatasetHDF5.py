import h5py
import numpy as np
import os
import cv2
import random

fullPathToDataset = \
    'C:/Users/62811/PCV/Big Project/CNN/DatasetH5PY'

folder = r"C:/Users/62811/PCV/Big Project/CNN/Dataset/Training/Images"
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
print(classes)

train = []
for i in classes:
    currentPath = os.path.join(folder, i)
    currentClass = classes.index(i)
    for j in os.listdir(currentPath):
        try:
            img = cv2.imread(os.path.join(folder, i, j))
            img = cv2.resize(img,(128,128))
            train.append([img, currentClass])
        except:
            continue

random.shuffle(train)
x = []
y = []
for i, j in train:
    x.append(i)
    y.append(j)

x_train = np.array(x) 
y_train = np.array(y)

print(x_train.shape)
print(y_train.shape)

x_temp = x_train[:int(x_train.shape[0] * 0.3), :, :, :]
y_temp = y_train[:int(y_train.shape[0] * 0.3)]

x_train = x_train[int(x_train.shape[0] * 0.3):, :, :, :]
y_train = y_train[int(y_train.shape[0] * 0.3):]

x_validation = x_temp[:int(x_temp.shape[0] * 0.8), :, :, :]
y_validation = y_temp[:int(y_temp.shape[0] * 0.8)]

x_test = x_temp[int(x_temp.shape[0] * 0.8):, :, :, :]
y_test = y_temp[int(y_temp.shape[0] * 0.8):]

os.chdir(fullPathToDataset)

with h5py.File('datasetCards.hdf5', 'w') as f:

    f.create_dataset('x_train', data=x_train, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    f.create_dataset('x_validation', data=x_validation, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    f.create_dataset('x_test', data=x_test, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')