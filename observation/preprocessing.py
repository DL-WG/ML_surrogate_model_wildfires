# -*- coding: utf-8 -*-
import numpy as np

import fiona

import geojson

from shapely.geometry import Polygon

import geopandas as gpd

import matplotlib.pyplot as plt



shapefile = gpd.read_file("CA/By_Fire/2017/Buck/Buck_2017_NAT.shp") #good

#shapefile = gpd.read_file("CA/By_Fire/2020/Bear/Bear_2020_NAT.geojson")

#shapefile = gpd.read_file("CA/By_Fire/2017/Pier/Pier_2017_NAT.shp")#good

print(shapefile.columns)

#x,y = np.array(shapefile)[0,11].exterior.coords.xy

x_list = []
        
y_list = []
    
for j in range(np.array(shapefile).shape[0]):
    
    geom = np.array(shapefile)[j,11]
    
    
    if geom.geom_type == 'Polygon':
        
        x,y = geom.exterior.coords.xy
        
        x_list += list(x)
        
        print(x)
        
        y_list += list(y)
        

        
    if geom.geom_type == 'MultiPolygon':
        
    
        mycoordslist = [list(x.exterior.coords) for x in geom.geoms]
        
        
        for i in range(len(mycoordslist)):
        
            x,y = zip(*mycoordslist[i])
        
            x_list += list(x)
            
            y_list += list(y)
            
############################################################"

#geom_ensemble = np.array(shapefile)[0,11]

x_burn = []
y_burn = []
    
for j in range(np.array(shapefile).shape[0]):

#for j in range(2):    
    geom = np.array(shapefile)[j,11]
    

    
    
    #geom_ensemble.union(geom)
    
    if geom.geom_type == 'Polygon':
        
        x,y = geom.exterior.coords.xy
        
        print(np.array(shapefile)[j,6])
        plt.fill(x,y,'r')
        #plt.title(np.array(shapefile)[j,6])
        
        plt.xlim([np.min(x_list),np.max(x_list)])
        plt.ylim([np.min(y_list),np.max(y_list)])
        
        #plt.title(np.array(shapefile)[j,6]) 
        
        for burn in range(len(x_burn)):
            
            plt.fill(x_burn[burn],y_burn[burn],'r',alpha = 0.0)
            
        
        plt.axis('off')
        #Jplt.savefig('figures/Pier_2017_'+str(j)+'.png', format='png')
        plt.savefig('fig_Buck/Buck_2017_'+str(j)+'.png', format='png')
        plt.show()
        plt.close()
        
        x_burn.append(x)
        y_burn.append(y)
        
    if geom.geom_type == 'MultiPolygon':
        
    
        mycoordslist = [list(x.exterior.coords) for x in geom.geoms]
        
        for i in range(len(mycoordslist)):
        
            x,y = zip(*mycoordslist[i])
        
            plt.fill(x,y,'r')
            
            plt.xlim([np.min(x_list),np.max(x_list)])
            
            plt.ylim([np.min(y_list),np.max(y_list)])
            
            plt.axis('off')
            
            x_burn.append(x)
            y_burn.append(y)
            #plt.savefig('ignition/Chimney_2016_initial.png', format='png')
            
        
        #plt.savefig('figures/Pier_2017_'+str(j)+'.png', format='png')
        for burn in range(len(x_burn)):
            
            plt.fill(x_burn[burn],y_burn[burn],'r',alpha = 0.)
        
        plt.savefig('fig_Buck/Buck_2017_'+str(j)+'.png', format='png')
        plt.show()
        plt.close()
            
        #plt.title(np.array(shapefile)[j,6]) 
    
        #plt.savefig('ignition/Ferguson_2018_initial.png', format='png')
        
        #plt.savefig('fig_Buck/Buck_2017_'+str(i)+'.png', format='png')
    #plt.show()
    print(np.array(shapefile)[j,6])
       
#    plt.close()
    
    
import pandas as pd
import pyproj


xcoords = [np.min(x_list), np.max(x_list)]
ycoords = [np.min(y_list), np.max(y_list)]
df = pd.DataFrame({'xcoords':xcoords, 'ycoords':ycoords})

#fips2401 = pyproj.Proj("+proj=tmerc +lat_0=35.83333333333334 +lon_0=-90.5 +k=0.9999333333333333 +x_0=250000 +y_0=0 +ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs")
fips2401=pyproj.Proj("+proj=aea +lat_1=34 +lat_2=40.5 +lat_0=0 +lon_0=-120 +x_0=0 +y_0=-4000000 +ellps=GRS80 +datum=NAD83 +units=m +no_defs ")
wgs84 = pyproj.Proj("+init=EPSG:4326")

df[['lon', 'lat']] = pd.DataFrame(pyproj.transform(fips2401, wgs84, df.xcoords.to_numpy(), df.ycoords.to_numpy())).T

print(df)