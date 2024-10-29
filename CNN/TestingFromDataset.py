import numpy as np
from keras.models import load_model
import h5py

with h5py.File('C:/Users/62811/PCV/Big Project/CNN/DatasetH5PY/datasetCardsGrayNormalized.hdf5', 'r') as f:

    x_test = f['x_test']  
    y_test = f['y_test']  

    x_test = np.array(x_test)  
    y_test = np.array(y_test)  
    
print('x_test:',x_test.shape)
print('y_test:',y_test.shape)

    
model = load_model ('C:/Users/62811/PCV/Big Project/CNN/DatasetH5PY/ALLINONE.h5')
predict = model.predict(x_test)
print('predict:',predict.shape)

predict = np.argmax(predict,axis=1)

print('predicted:',predict[0:10])
print('correct:',y_test[0:10])

accuracy=np.mean(predict==y_test)
print('True or False',(predict == y_test)[0:10])
print('accuracy:{0:.3f}'.format(accuracy))