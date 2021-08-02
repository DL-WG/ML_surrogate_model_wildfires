# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib as mpl


field = np.loadtxt('fields/Pier_2017_'+str(15)+'_'+str(350)+'.txt')

number_MC = 20
generation = 500
fields_1_sim = np.zeros((1,field.shape[0]*field.shape[1]))

for index in range(0,number_MC):        
    
    for i in range(0,generation,1):
        
        field = np.loadtxt('fields/Pier_shift_800_2017_'+str(index)+'_'+str(i)+'.txt')
        fields_1_sim = np.concatenate((fields_1_sim,
                                       field.reshape(1,field.size)),axis = 0)
        

fields_1_sim = fields_1_sim[1:,]
u_pod, s_pod, v_pod = np.linalg.svd(fields_1_sim.T, full_matrices=False)

np.savetxt("data/u_pod_Pier_2017.txt",u_pod[:,:100])
