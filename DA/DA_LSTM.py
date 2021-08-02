# -*- coding: utf-8 -*-
import sys
import math
import random
import copy
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import animation as animation

from PIL import Image
import matplotlib as mpl
#import imageio
import pickle
import tensorflow as tf
from tensorflow import keras
import time
import scipy
import time
from DA_preparation import *
from constructB import *
from tensorflow import keras

number_MC = 1 #number of simulated trajectories

generation = 500 #number of simulation time steps in total

prediction_start = 20

#########################################################
with_DA = 1
DI01_it = 5 #number of DI01 iterations
day_step = 40
########################################################


# custormize colorbar

cmap = mpl.colors.ListedColormap(['orange','yellow', 'green', 'red'])
cmap.set_over('0.25')
cmap.set_under('0.75')
bounds = [1.0, 2.03, 2.35, 3.5, 5.1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

############################################################

forest = Image.open('images/canopy_Bear_2020.tif')

ignition = np.loadtxt('ignitions/Bear_2020_ignition_forest.txt')

altitude = Image.open('images/slope_Bear_2020.tif')

density = Image.open('images/density_Bear_2020.tif')

###########################################################
###########################################################

forest = np.array(forest)

altitude = np.array(altitude)
 
density = np.array(density)

density = np.round(density*2./np.max(density))

forest[forest<-999.] = 0.

forest = forest/np.max(forest)

n_row = forest.shape[0]
n_col = forest.shape[1]

#################################################################
# load LSTM model

filename = 'ML_model/LSTM_Pier_2017_sequence_20_POD_100_double_until200_modeladvanced_newbis'

LSTM_POD_model = keras.models.load_model(filename)

#############################################################
#initializing variables

POD_series = np.zeros((20,100)) # sequence of input

POD_predict = np.zeros((20,100))# sequence of output prediction

DL_error = []

pod_error_MC = []

time_LSTM = []

norm_field = []

prediction_error_vect = np.zeros((10,12))

DA_error_vect = np.zeros((10,12))

############################################################

for index in range(0,0+number_MC):
    
    
    LSTM_error = []
    
    prediction_error = []
    
    obspod_error = []
    
    DA_error = []

    print('index',index)
    
    for i in range(0,generation):
        
        
        field = np.loadtxt('fields/Bear_2020_'+str(index)+'_'+str(i)+'.txt')
        
        field_initial = np.copy(field)
        
        norm_field.append(np.linalg.norm(field))  

        
        
##################################################################################
        #read observations: y
        #reconstructing osberved fires
        if i%day_step == 0:
                    
            obs_fire = np.loadtxt('observation_fire/Bear_2020_fire_'+str(int(i/day_step))+'.txt')
            
            obs_burn = np.loadtxt('observations/Bear_2020_'+str(int(i/day_step))+'.txt')
            
            obs_field = np.ones((n_row,n_col))*2 + obs_burn*2 - obs_fire
            
            
            obs_field[obs_field>=3] =4

            im = plt.imshow(forest + np.round( obs_field.reshape(n_row,n_col)), cmap = cmap,norm = norm,
                            interpolation = 'None', vmin = 0., vmax = 4. )
            plt.axis('off')

            plt.show()
##################################################################################"
        
        if i<prediction_start:
            
            #########################################################################
            #LSTM
            #########################################################################
          
            POD_series.shape = (20,100)
            
            POD_series = np.roll(POD_series, -1, axis=0)
            
            
            field_initial.shape = (field_initial.size,1)
            
            POD_initial = np.dot (u_pod[:,:100].T,field_initial)

            POD_series[19,:] = POD_initial.ravel()
            
            POD_initial.shape = (POD_initial.size,1)
            
            field_reconstruction = np.round(np.dot(u_pod[:,:100], POD_initial))#state: x
            
            
            #####################################################################           
        else:
            if (i-prediction_start)%20 == 0:

                                                  
                POD_series.shape = (1,20,100)
                
                t = time.time()
                
                POD_p = LSTM_POD_model.predict(POD_series)
                
                time_LSTM.append(time.time()-t)
                
                POD_p.shape = (20,100)
                
            #####################################################################
            index_in_p = ((i-prediction_start)%20)
            
            POD_initial_predict = POD_p[index_in_p ,:]
            
            ######################################################################
            # DA
            if with_DA == 1:
                
                xb = np.copy(POD_initial_predict)
                
                if i % day_step <= 20:
                    
                    obs_field = np.ones((n_row,n_col))*2 + obs_burn*2 - obs_fire
    #                     
                    y = np.dot (u_pod[:,:100].T,obs_field.reshape(obs_field.size,1))
                    
                
                    B = 1*np.eye(100)
                                        
                    R = 1*Balgovind_1D(100,15)
                                        
                    H_DA = np.eye(100)
                                                                                                                            
                    for iteration in range(DI01_it):
    
                        t = time.time()
                        xa,_,B,R,sb,so = DI01(xb.reshape(xb.size,1),y.reshape(y.size,1),H_DA,B,R)

                    POD_initial_predict = np.copy(xa)
                    
                    
            #########################################################################
            
            #########################################################################
            
            POD_initial_predict.shape = (POD_initial_predict.size,1)
            
            field_reconstruction_predict = np.dot(u_pod[:,:100], POD_initial_predict)
            
                
            if i % day_step == 0:
                
                obs_field = np.ones((n_row,n_col))*2 + obs_burn*2 - obs_fire
                
                y_round = np.round(np.copy(obs_field))
                    
                y_round[y_round>=4] = 4
                
                x = np.round(np.copy(field_reconstruction_predict))

                if with_DA == 1:
                                                   
                    field_reconstruction_background = np.dot(u_pod[:,:100], xb.reshape(xb.size,1))

                    

                    field_reconstruction_background = np.dot(u_pod[:,:100], xb.reshape(xb.size,1))
                    
                    field_reconstruction_background = np.round(np.dot(u_pod[:,:100], xb.reshape(xb.size,1)))
                    
                    field_reconstruction_background[field_reconstruction_background>=4] =4
                                   
                    print('xb',(np.linalg.norm(field_reconstruction_background.ravel()-y_round.ravel())/np.linalg.norm(y_round.ravel())))

                    obs_reconstructed = np.round(np.dot(u_pod[:,:100],np.dot(u_pod[:,:100].T,y_round.reshape(y_round.size,1))))
                    
                    obspod_error.append(np.linalg.norm(obs_reconstructed.ravel()-y_round.ravel())/np.linalg.norm(y_round.ravel()))
                    
                    
                    
                    obs_reconstructed[obs_reconstructed>=3] =4
                
                    print('obs reconstructed')
                           
                    
                    prediction_error.append(np.linalg.norm(field_reconstruction_background.ravel()-y_round.ravel())
                    /np.linalg.norm(y_round.ravel()))  
                    
                    field_reconstruction_background[field_reconstruction_background>=3] =4
                    
                    im = plt.imshow(forest + np.round( field_reconstruction_background.reshape(n_row,n_col)), cmap = cmap,norm = norm,
                                    interpolation = 'None', vmin = 0., vmax = 4. )
                    plt.axis('off')

                    plt.show(im)

                    plt.close()                    
          #####################################################################

            if i%day_step == 0:
                
                obs_field = np.ones((n_row,n_col))*2 + obs_burn*2 - obs_fire
                               
                y_round = np.round(np.copy(obs_field))
                
                field_reconstruction_predict = np.round(field_reconstruction_predict)
                
                field_reconstruction_predict[field_reconstruction_predict>=4] =4
                              
                print('xa',np.linalg.norm(field_reconstruction_predict.ravel()-y_round.ravel())/np.linalg.norm(y_round.ravel()))
                
                DA_error.append(np.linalg.norm(field_reconstruction_predict.ravel()-y_round.ravel())/np.linalg.norm(y_round.ravel())) 
          
                field_reconstruction_predict[field_reconstruction_predict>=3] =4
                
                
                im = plt.imshow(forest + np.round( field_reconstruction_predict.reshape(n_row,n_col)), interpolation = 'None',norm = norm,
                                cmap = cmap, vmin = 0., vmax = 4. )
                plt.axis('off')
                plt.show()
                plt.close()
                
                print("============================================================================================")
            
            POD_series.shape = (1,20,100)
                        
            POD_series = np.roll(POD_series, -1, axis=0)
            
            POD_series[0,19,:] = POD_initial_predict.ravel()
            
            POD_series.shape = (20,100)
            

    if with_DA == 1:
        prediction_error_vect[index-10,:] = np.array(prediction_error)
        
        obspod_error_vect[index-10,:] = np.array(obspod_error)
    
    DA_error_vect[index-10,:] = np.array(DA_error)
