# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 05:01:05 2020

@author: Fahman Saeed
"""



# In[ ]:

# import libraries
import os
import glob
import numpy as np
import pandas as pd 
import tensorflow 
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from scipy.io import loadmat
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.models import Sequential
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.layers import TimeDistributed,LSTM,Dense, Flatten, Conv2D, MaxPooling2D, Dropout,Activation,BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D,concatenate, SpatialDropout1D, TimeDistributed, Bidirectional, LSTM
from tensorflow.keras import optimizers, losses, activations, models
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
tf.keras.backend.clear_session()
import matplotlib.pyplot as plt



#=========================================================================== 
#                       Functions
#===========================================================================          
#shuffle_data(data): shuffle data
def shuffle_data(data):
  data=shuffle(data)
  return data

# load_samples
def load_data(path):
    X=[];y=[];samples=[]
    for path, subdirs, files in os.walk(path):
        source1=path
        for name in files:
            c=os.path.split(path)[1]
            img=loadmat(os.path.join(path, name))['NewChunk']
            X.append(img.reshape(-1))
            # X.append(img)
            y.append(c)
        
    return X,y

data_path='Data'#<-------------- change this regarding to the new data path 
X,y = load_data(data_path)
X=np.asarray(X)


#%% 
#parameters
batch_size=5;epochs = 10;img_cols, img_rows=32,1280

#standarize data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
x = scaler.transform(X)
y=[int(i) for i in y]
nc=len(np.unique(y))
y1 = to_categorical(y)
y=y1[:,1:11]# remove index 0 column
# devided data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size = 0.1, random_state = 4)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = 4)

# rehsape data
x_train = x_train.reshape(x_train.shape[0], img_cols, img_rows,1)
x_valid = x_valid.reshape(x_valid.shape[0], img_cols, img_rows,1)
x_test = x_test.reshape(x_test.shape[0], img_cols, img_rows,1)
# show data dim
print("Training samples: ", len(x_train),'train shape: ',x_train.shape)
print("Validation samples: " ,len(x_valid))
print("Testing samples: " ,len(x_test))
# build data using Dataset container
train_dataset =tf.data.Dataset.from_tensor_slices((x_train,y_train)).repeat().batch(batch_size)
valid_dataset =tf.data.Dataset.from_tensor_slices((x_valid,y_valid)).repeat().batch(batch_size)
test_dataset =tf.data.Dataset.from_tensor_slices((x_test,y_test)).repeat().batch(batch_size)
# train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
#%%  create model  CNN and LSTM
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, activation='relu',input_shape=(  img_cols,img_rows,1)))
# model.add(tf.keras.layers.Conv2D(32, (3, 3)))
model.add(Conv2D(32, (3, 3)))
model.add(Conv2D(64, (3, 3)))
model.add(tf.keras.layers.Reshape((-1, 64)))
model.add(LSTM(64, input_shape=(40960,1),activation="relu",return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32,activation="sigmoid"))
model.add(Dropout(0.5))
model.add(Dense(nc, activation='sigmoid'))
model.compile(loss = 'categorical_crossentropy', optimizer = "RMSprop", metrics = ['accuracy'])

#%% train model
history = model.fit(train_dataset,
                    epochs=epochs, 
                    steps_per_epoch=5,
                    validation_steps=5,
                    validation_data=valid_dataset)
score, acc = model.evaluate(x_test, y_test)
print('test score: ',score,' acc: ',acc)

#%% prediction
from sklearn.metrics import accuracy_score,classification_report
pred = model.predict(x_test)
predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y_test,axis=1)
print(expected_classes.shape)
print(predict_classes.shape)
correct = accuracy_score(expected_classes,predict_classes)
print(f"Training Accuracy: {correct}")
print(classification_report(expected_classes,predict_classes))
#%% plot perfomance you can use acc or accuracy based on version of TF ; also val_acc val_accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(int(epochs))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
