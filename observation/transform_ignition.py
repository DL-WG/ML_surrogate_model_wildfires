# -*- coding: utf-8 -*-
import numpy as np

import fiona

import geojson

from shapely.geometry import Polygon

import geopandas as gpd

import matplotlib.pyplot as plt

shapefile = gpd.read_file("CA/By_Fire/2020/Bear/Bear_2020_NAT.geojson")

#################################################################################

forest = np.array(forest)

n_row = forest.shape[0]
n_col = forest.shape[1]

n_row = 882
n_col = 838

img = Image.open('ignition/Bear_2020_initial.png')

img = img.resize((n_row,n_col), Image.ANTIALIAS)

img_matrix = np.asarray(img)[:,:,1]

im = plt.imshow(img_matrix)
plt.colorbar(im)
plt.show()

ignition = np.copy(img_matrix)

ignition[ignition<=1.] = 3

ignition[ignition==255.] = 2

ignition[ignition<=1.5] = 2

ignition[ignition>=2.5] = 3

im = plt.imshow(ignition)
plt.colorbar(im)
plt.show()

np.savetxt('ignition/Bear_2020_ignition_forest.txt',ignition)

