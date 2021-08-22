# -*- coding: utf-8 -*-
# preprocessing for forming the training/testing dataset

import numpy as np
import scipy
import math
import matplotlib.pyplot as plt

# total time steps of fire simulations
generation = 500


for index in range(0,120):
    fields_1_sim = np.zeros((1,100))#100 is the dimension of the latent space
    print('index',index)
    for i in range(generation):

        
        field_pod = np.loadtxt('POD_MC/Bear_2020_POD_obs_'+str(index)+'_'+str(i)+'.txt')



        fields_1_sim = np.concatenate((fields_1_sim,field_pod.reshape(1,field_pod.size)),axis = 0)
      
    fields_1_sim = fields_1_sim[1:,:]
    #print(fields_1_sim.shape)
    
    np.savetxt('POD_MC/Bear_2020_POD_obs_allsim_'+str(index)+'.txt',fields_1_sim)


n_memory = 20 #number of look back steps

n_prediction = 20 #number of prediction steps

latent_dim = 100

input_data = np.zeros((1,n_memory,latent_dim))
output_data = np.zeros((1,n_prediction,latent_dim))



for index in range(60,100):
    
    print('index',index)
    fields_1_sim = np.loadtxt('POD_MC/Bear_2020_POD_allsim_'+str(index)+'.txt')
    
    current_simulation = np.copy(fields_1_sim)
    #    


    for j in range(500-n_memory-n_prediction):

        input_lstm = current_simulation[j:(j+n_memory),:]

        input_data = np.concatenate((input_data,input_lstm.reshape(1,n_memory,latent_dim)),axis = 0)

        output_lstm = current_simulation[(j+n_memory):(j+n_memory+n_prediction),:]

        output_data = np.concatenate((output_data,output_lstm.reshape(1,n_prediction,latent_dim)),axis = 0)
        
    print('input_data.shape',input_data.shape)
        
    print('output_data.shape',output_data.shape)
    
input_data = input_data[1:,:]
output_data = output_data[1:,:]

np.save('LSTM_data/Bear_encoder_2020_input_window20.npy',input_data)
np.save('LSTM_data/Bear_encoder_2020_output_window20.npy',output_data)

