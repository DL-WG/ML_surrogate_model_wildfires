# -*- coding: utf-8 -*-
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt

from tensorflow.python.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Activation,Dropout,RepeatVector
from tensorflow.python.keras.layers import BatchNormalization, TimeDistributed
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint

from tensorflow.python.keras import backend as K

import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.keras import optimizers
# check scikit-learn version

# check scikit-learn version
import pandas as pd

import os
import json
import pickle



################################################################
#define input and output sequence size
n_memory = 20
n_prediction = 20

input_data = np.load('LSTM_data/Bear_POD_2017_input_window20_0_120.npy')

output_data = np.load('LSTM_data/Bear_POD_2017_output_window20_0_120.npy')

###############################################################################
#preprocessing 

#########################################################################
train_part = 0.9

threshold = int(train_part*input_data.shape[0])


##########################################################################

train_input = input_data[:threshold,:,:]

train_output = output_data[:threshold,:]

test_input = input_data[threshold:,:,:]

true_test_output = output_data[threshold:,:]

X1 = train_input
Y1 = train_output

X2 = test_input

#######################################################################
hidden_size=200

input_sample = input_data.shape[0]  #for one sample

output_sample = output_data.shape[0]

use_dropout=True

model = Sequential()

model.add(LSTM(hidden_size,input_shape=(20,100)))
#model.add(Dropout(0.3))

model.add(RepeatVector(n_prediction))

#multi-step
model.add(LSTM(100, activation='relu', return_sequences=True))


model.add(Dense(100))

model.add(Dense(100))


model.add(TimeDistributed(Dense(100)))

model.add(Activation('relu'))

##################################################################
#training

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
history = model.fit(input_data, output_data, validation_split=0.1, epochs=80,batch_size=64, verbose=1)

####################################################################
# evalutation in the latent space

PredTestSet = model.predict(X1)
PredValSet = model.predict(X2)

plt.plot(PredValSet[:,0,0],true_test_output[:,0,0],'o', color='blue',markersize=5)
plt.plot(PredValSet[:,0,0],PredValSet[:,0,0], color='r',markersize=5)
#plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)

plt.show()



plt.plot(PredValSet[:,0,1],true_test_output[:,0,1],'o', color='blue',markersize=5)
plt.plot(PredValSet[:,0,1],PredValSet[:,0,1], color='r',markersize=5)
#plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)

plt.show()

plt.plot(PredValSet[:,0,2],true_test_output[:,0,2],'o', color='blue',markersize=5)
plt.plot(PredValSet[:,0,2],PredValSet[:,0,2], color='r',markersize=5)
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)

plt.show()

plt.plot(PredValSet[:,0,99],true_test_output[:,0,99],'o', color='blue',markersize=5)
plt.plot(PredValSet[:,0,99],PredValSet[:,0,99], color='r',markersize=5)
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)

plt.show()

####################################################################""

#plt.plot(PredValSet[:,0,3],true_test_output[:,0,3],'o', color='blue',markersize=5)
#plt.plot(PredValSet[:,0,3],PredValSet[:,3], color='r',markersize=5)
#plt.plot(list(range(0,1,0.1)),list(range(0,1,0.1)),'k')
#plt.show()
plt.plot(PredValSet[:,10,0],true_test_output[:,10,0],'o',color='blue',markersize=5)
plt.plot(PredValSet[:,10,0],PredValSet[:,10,0], color='r',markersize=5)
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)

plt.show()

plt.plot(PredValSet[:,10,1],true_test_output[:,10,1],'o',color='blue',markersize=5)
plt.plot(PredValSet[:,10,1],PredValSet[:,10,1], color='r',markersize=5)
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)

plt.show()

plt.plot(PredValSet[:,10,2],true_test_output[:,10,2],'o',color='blue',markersize=5)
plt.plot(PredValSet[:,10,2],PredValSet[:,10,2], color='r',markersize=5)
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)

plt.show()

plt.plot(PredValSet[:,10,99],true_test_output[:,10,99],'o',color='blue',markersize=5)
plt.plot(PredValSet[:,10,99],PredValSet[:,10,99], color='r',markersize=5)
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)

plt.show()
#
########################################################################""
plt.plot(PredValSet[:,19,0],true_test_output[:,19,0],'o',color='blue',markersize=5)
plt.plot(PredValSet[:,19,0],PredValSet[:,19,0], color='r',markersize=5)
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)

plt.show()

plt.plot(PredValSet[:,19,1],true_test_output[:,19,1],'o',color='blue',markersize=5)
plt.plot(PredValSet[:,19,1],PredValSet[:,19,1], color='r',markersize=5)
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)

plt.show()

plt.plot(PredValSet[:,19,2],true_test_output[:,19,2],'o',color='blue',markersize=5)
plt.plot(PredValSet[:,19,2],PredValSet[:,19,2], color='r',markersize=5)
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)

plt.show()

plt.plot(PredValSet[:,19,99],true_test_output[:,19,99],'o',color='blue',markersize=5)
plt.plot(PredValSet[:,19,99],PredValSet[:,19,99], color='r',markersize=5)
plt.xlabel('prediction',fontsize = 16)
plt.ylabel('true value',fontsize = 16)

plt.show()
#
