# -*- coding: utf-8 -*-
from tensorflow.python.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Cropping2D
from tensorflow.python.keras.models import Model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle


field = np.loadtxt('fields/Bear_2020_'+str(10)+'_'+str(0)+'.txt')
print(field.shape)

n_row,n_col = field.shape

generation = 500

sim_num = 10

field_ensemble = np.zeros((generation*2+10,805, 749))

j = 0

################################################################
#construct training dataset

for index in range(0,10):
    for i in range(generation):
        
        if i%5 == 0:

            print(index,i)
            field = np.loadtxt('fields/Bear_2020_'+str(index)+''+str(i)+'.txt')
    
            field_ensemble[j,:,:] = field
        
            j += 1
##################################################################
# preprocessing

field_ensemble[field_ensemble <= 2.] = 0.

field_ensemble[field_ensemble == 3] = 0.5

field_ensemble[field_ensemble == 4.] = 1.

train_part = 0.9

threshold = int(train_part*field_ensemble.shape[0])

x_train = field_ensemble[:threshold]

x_test = field_ensemble[threshold:]


x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
################################################################
# CNN encoder

x_train = np.reshape(x_train, (len(x_train), n_row,n_col, 1))
x_test = np.reshape(x_test, (len(x_test), n_row,n_col, 1))
print('---> xtrain shape: ', x_train.shape)
print('---> x_test shape: ', x_test.shape)

input_img = Input(shape=(n_row,n_col, 1))
 
x = Convolution2D(8, (10, 10), activation='relu', padding='same')(input_img)
x = MaxPooling2D((5, 5), padding='same')(x)
x = Convolution2D(4, (2, 2), activation='relu', padding='same')(x)
x = MaxPooling2D((3, 3), padding='same')(x)
x = Convolution2D(4, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(1, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

encoder = Model(input_img, encoded)


decoder_input= Input(shape=(14,13,1))

decoder = Convolution2D(1, (2, 2), activation='relu', padding='same')(decoder_input)

x = UpSampling2D((2, 2))(decoder)
#x = ZeroPadding2D(((0,1),(1,1)))(x)
x = Convolution2D(4, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((3, 3))(x)
x = Convolution2D(4, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((5, 5))(x)
x = Convolution2D(8, (10, 10), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Cropping2D(cropping=((35, 0), (31, 0)), data_format=None)(x)
#decoded = Cropping2D(cropping=((5, 0), (1, 0)), data_format=None)(x)
decoded = Convolution2D(1, (10, 10), activation='sigmoid', padding='same')(x)

####################################################################
#combine encoder and decoder 

auto_input = Input(shape=( n_row,n_col,1))
encoded = encoder(auto_input)
decoded = decoder(encoded)

autoencoder = Model(auto_input, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

####################################################################

history = autoencoder.fit(x_train, x_train, epochs=200, batch_size=8,shuffle=True, validation_data=(x_test, x_test))