# -*- coding: utf-8 -*-

import numpy as np

import fiona

import geojson

from shapely.geometry import Polygon

import geopandas as gpd

import matplotlib.pyplot as plt

from PIL import Image

n_row = 838
n_col = 882

day = 20

accumulate = np.zeros((n_row,n_col))

for j in range(0,day+1):

    img = Image.open('figures/Pier_2017_'+str(j)+'.png')
    
    img = img.resize((n_col,n_row), Image.ANTIALIAS)
    
    img_matrix = np.asarray(img)[:,:,1]
    
    state = np.copy(img_matrix)
#
#    
    state[state<=1.5] = 1
##    
    state[state==255.] = 0.
    
    state[state>=1.] = 1
    
    accumulate += state
    
    accumulate[accumulate >= 1.] = 1.

    np.savetxt('observation_fire/Pier_2017_fire_'+str(j)+'.txt',state)
    
    #np.savetxt('observations/Pier_2017_'+str(j)+'.txt',accumulate)
    
    #im = plt.imshow(accumulate)
    
    im = plt.imshow(state)
    
    plt.colorbar(im)
    
    plt.show()
    
########################################################
    
all_index = list(range(0,500))
train_index = []
for i in range(0,461,5):
    train_index += list(range(i,i+4))   

test_index = list(set(all_index)-set(train_index ))

