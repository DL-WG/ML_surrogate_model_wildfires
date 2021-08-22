# -*- coding: utf-8 -*-
#In this script we demonstrate the CA simulation using the data of the Bear fire in 2020

import sys
import math
import random
import copy
#from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import animation as animation

from PIL import Image
import matplotlib as mpl

##################################################################
#loading data

#test with real map

forest = Image.open('images/canopy_Bear_2020.tif')

ignition = np.loadtxt('ignitions/Bear_2020_ignition_forest.txt')

altitude = Image.open('images/slope_Bear_2020.tif')

#fuel = Image.open('images/fuel_Ferguson_2018.tif')

density = Image.open('images/density_Bear_2020.tif')

####################################################################
#preprocessing 


forest = np.array(forest)

altitude = np.array(altitude)
 
#fuel = np.array(fuel )

density = np.array(density)

density = np.round(density*2./np.max(density))

forest[forest<-999.] = 0.

forest = forest/np.max(forest)


n_row = forest.shape[0]
n_col = forest.shape[1]

number_MC = 20
generation = 500

#################################################################
#definition of the CA simulator

def colormap(i,array):
    np_array = np.array(array)
    plt.imshow(np_array, interpolation="none", cmap=cm.plasma)
    plt.title(i)
    plt.show()


def init_vegetation():
    veg_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
    for i in range(n_row):
        for j in range(n_col):
            veg_matrix[i][j] = 1
    return veg_matrix


def init_density():
    den_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
    for i in range(n_row):
        for j in range(n_col):
            den_matrix[i][j] = 1.0
    return den_matrix


def init_altitude():
    alt_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
    for i in range(n_row):
        for j in range(n_col):
            alt_matrix[i][j] = 1
    return alt_matrix


def init_forest():
    forest = [[0 for col in range(n_col)] for row in range(n_row)]
    for i in range(n_row):
        for j in range(n_col):
            forest[i][j] = 2
    # ignite_col = int(n_col//2)
    # ignite_row = int(n_row//2)
    ignite_col = int(n_col//2)
    ignite_row = int(100)
    for row in range(ignite_row-1, ignite_row+1):
        for col in range(ignite_col-1,ignite_col+1):
            forest[row][col] = 3
    # forest[ignite_row-2:ignite_row+2][ignite_col-2:ignite_col+2] = 3
    return forest


def print_forest(forest):
    for i in range(n_row):
        for j in range(n_col):
            sys.stdout.write(str(forest[i][j]))
        sys.stdout.write("\n")


def tg(x):
    return math.degrees(math.atan(x))


def get_slope(altitude_matrix):
    slope_matrix = [[0 for col in range(n_col)] for row in range(n_row)]
    for row in range(n_row):
        for col in range(n_col):
            sub_slope_matrix = [[0,0,0],[0,0,0],[0,0,0]]
            if row == 0 or row == n_row-1 or col == 0 or col == n_col-1:  # margin is flat
                slope_matrix[row][col] = sub_slope_matrix
                continue
            current_altitude = altitude_matrix[row][col]
            sub_slope_matrix[0][0] = tg((current_altitude - altitude_matrix[row-1][col-1])/1.414)
            sub_slope_matrix[0][1] = tg(current_altitude - altitude_matrix[row-1][col])
            sub_slope_matrix[0][2] = tg((current_altitude - altitude_matrix[row-1][col+1])/1.414)
            sub_slope_matrix[1][0] = tg(current_altitude - altitude_matrix[row][col-1])
            sub_slope_matrix[1][1] = 0
            sub_slope_matrix[1][2] = tg(current_altitude - altitude_matrix[row][col+1])
            sub_slope_matrix[2][0] = tg((current_altitude - altitude_matrix[row+1][col-1])/1.414)
            sub_slope_matrix[2][1] = tg(current_altitude - altitude_matrix[row+1][col])
            sub_slope_matrix[2][2] = tg((current_altitude - altitude_matrix[row+1][col+1])/1.414)
            slope_matrix[row][col] = sub_slope_matrix
    return slope_matrix


def calc_pw(theta):
    c_1 = 0.045
    c_2 = 0.131
    V = 10
    t = math.radians(theta)
    ft = math.exp(V*c_2*(math.cos(t)-1))
    return math.exp(c_1*V)*ft


def get_wind():

    wind_matrix = [[0 for col in [0,1,2]] for row in [0,1,2]]
#    thetas = [[45,0,45],
#              [90,0,90],
#              [135,180,135]]
    thetas = [[0,0,0],
              [0,0,0],
              [0,0,0]]
#    thetas = [[135,90,45],
#              [180,0,0],
#              [135,90,45]] 
    
    for row in [0,1,2]:
        for col in [0,1,2]:
            wind_matrix[row][col] = calc_pw(thetas[row][col])
    wind_matrix[1][1] = 0
    return wind_matrix


def burn_or_not_burn(abs_row,abs_col,neighbour_matrix):
    p_veg = vegetation_matrix[abs_row][abs_col]
    #p_den = {1:-0.4,2:0,3:0.3}[density_matrix[abs_row][abs_col]]
    p_den = {0:-0.4,1:0,2:0.3}[density_matrix[abs_row][abs_col]]
    p_h = 0.58
    a = 0.078

    for row in [0,1,2]:
        for col in [0,1,2]:
            if neighbour_matrix[row][col] == 3: # we only care there is a neighbour that is burning
                # print(row,col)
                slope = slope_matrix[abs_row][abs_col][row][col]
                p_slope = math.exp(a * slope)
                p_wind = wind_matrix[row][col]
                p_burn = p_h * (0.5 + p_veg*10.) * (1 + p_den) * p_wind * p_slope
                if p_burn > random.random():
                    return 3  #start burning

    return 2 # not burning


def update_forest(old_forest):
    result_forest = [[1 for i in range(n_col)] for j in range(n_row)]
    for row in range(1, n_row-1):
        for col in range(1, n_col-1):

            if old_forest[row][col] == 1 or old_forest[row][col] == 4:
                result_forest[row][col] = old_forest[row][col]  # no fuel or burnt down
            if old_forest[row][col] == 3:
                if random.random() < 0.4:
                    result_forest[row][col] = 3  # TODO need to change back here
                else:
                    result_forest[row][col] = 4
            if old_forest[row][col] == 2:
                neighbours = [[row_vec[col_vec] for col_vec in range(col-1, col+2)]
                              for row_vec in old_forest[row-1:row+2]]
                # print(neighbours)
                result_forest[row][col] = burn_or_not_burn(row, col, neighbours)
    return result_forest

##################################################################################
# MC simulations with same fire scenario

for index in range(20,21):
    

    
    print ('index',index)

    vegetation_matrix = forest
        
    density_matrix = density.tolist()
   
    altitude_matrix = altitude.tolist()
    
    wind_matrix = get_wind()
    
    new_forest = ignition.tolist()
        
    slope_matrix = get_slope(altitude_matrix)
    
    ims = []
    
    ###########################################################
    # custormize colorbar
   
    for i in range(generation):
        
        print(index,i)
        new_forest = copy.deepcopy(update_forest(new_forest))
    
        
        forest_array = np.array(new_forest)
                  
        plt.imshow(forest_array.reshape(n_row,n_col))       
        plt.savefig('figure_gif/fire_total_'+ str(i) +'.png', format='png')
        plt.show()
        plt.close()
