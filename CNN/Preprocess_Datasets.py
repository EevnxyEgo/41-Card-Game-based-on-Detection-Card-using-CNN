
import numpy as np
import h5py
import cv2
import os

fullPathToDataset = \
    'C:/Users/62811/PCV/Big Project/CNN/DatasetH5PY'

with h5py.File('C:/Users/62811/PCV/Big Project/CNN//DatasetH5PY/datasetCards.hdf5', 'r') as f:

    x_train = f['x_train']  
    y_train = f['y_train']  

    x_train = np.array(x_train)  
    y_train = np.array(y_train)  

    x_validation = f['x_validation']  
    y_validation = f['y_validation']  

    x_validation = np.array(x_validation)  
    y_validation = np.array(y_validation)  

    x_test = f['x_test']  
    y_test = f['y_test']  

    x_test = np.array(x_test)  
    y_test = np.array(y_test) 
    
#preproses

x_train = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), x_train)))

#lambda sama kayak
#def cvtGray(x):
    # img = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    # return img
#x_train = np.array(list(map(cvtGray,x_train)))
    
x_validation = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), x_validation)))
x_test = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), x_test)))

x_train = x_train[:, :, :, np.newaxis]
x_validation = x_validation[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]

print(x_train.shape)
print(x_validation.shape)
print(x_test.shape)

x_train_255 = x_train / 255.0
x_validation_255 = x_validation / 255.0
x_test_255 = x_test / 255.0

os.chdir(fullPathToDataset)

with h5py.File('datasetCardsGrayNormalized.hdf5', 'w') as f:
    f.create_dataset('x_train', data=x_train_255, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    f.create_dataset('x_validation', data=x_validation_255, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    f.create_dataset('x_test', data=x_test_255, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')



