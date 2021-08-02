# coding: utf8
#construction of matrix B and special H with measure on the boarder
import numpy as np
import math
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

##def B_Balgovind(n,Sigma,L):
##    Gamma = np.identity(n)
##    for i in xrange(n):
##        for j in xrange(n):
##            Gamma[i,j] = ( 1. + abs(i-j)/L)*np.exp(-abs(i-j)/L)
##    B = np.dot(Sigma,np.dot(Gamma,Sigma))
##    return B


def get_index_2d (dim,n): #get caratesian coordinate
    j=n % dim
    j=j/1. #float( i)
    i=(n-j)/dim
    return (i,j)# pourquoi float?

def get_index_2d_general (dim1,dim2,n): #get caratesian coordinate
    j=n % dim2
    j=j/1. #float( i)
    i=(n-j)/dim1
    return (i,j)# pourquoi float?

#identite
def identiity(n):#n : taille de vecteur xb

    B=np.eye(n)
    return B

#Blgovind
def Balgovind(dim,L):

    L = L*1.
    sub_B=np.zeros((dim**2,dim**2))
    for i in range(dim**2):
        (a1,b1)=get_index_2d(dim,i)
        for j in range(dim**2):
            (a2,b2)=get_index_2d(dim,j) #reprends les donnees caracterisennes
            r = math.sqrt((a1-a2)**2+(b1-b2)**2)*1.

            sub_B[i,j]=(1.+r/L)*(math.exp(-r/L))

                                
                
    B1=np.concatenate((sub_B, np.zeros((dim**2,dim**2))), axis=1)
    B2=np.concatenate(( np.zeros((dim**2,dim**2)), sub_B),axis=1)
    B=np.concatenate((B1,B2), axis=0)# a changer construction matrice B
    return B


def Balgovind_general(dim1,dim2,L):

    L = L*1.
    sub_B=np.zeros((dim1*dim2,dim1*dim2))
    for i in range(dim1*dim2):
        (a1,b1)=get_index_2d_general(dim1,dim2,i)
        for j in range(dim1*dim2):
            (a2,b2)=get_index_2d_general(dim1,dim2,j) #reprends les donnees caracterisennes
            r = math.sqrt((a1-a2)**2+(b1-b2)**2)*1.

            sub_B[i,j]=(1.+r/L)*(math.exp(-r/L))

                                
                
    return sub_B

def Gaussian(dim,L):
    sub_B=np.zeros((dim**2,dim**2))
    for i in range(dim**2):
        (a1,b1)=get_index_2d(dim,i)
        for j in range(dim**2):
            (a2,b2)=get_index_2d(dim,j) #reprends les donnees caracterisennes
            r=math.sqrt((a1-a2)**2+(b1-b2)**2)
            sub_B[i,j]=math.exp(-r**2/(2*L**2))
                                
                
    B1=np.concatenate((sub_B, np.zeros((dim**2,dim**2))), axis=1)
    B2=np.concatenate(( np.zeros((dim**2,dim**2)), sub_B),axis=1)
    B=np.concatenate((B1,B2), axis=0)# a changer construction matrice B
    return B

def expontielle(dim,L):
    sub_B=np.zeros((dim**2,dim**2))
    for i in range(dim**2):
        (a1,b1)=get_index_2d(dim,i)
        for j in range(dim**2):
            (a2,b2)=get_index_2d(dim,j) #reprends les donnees caracterisennes
            r=math.sqrt((a1-a2)**2+(b1-b2)**2)
            sub_B[i,j]=math.exp(-r/L)
                                
                
    B1=np.concatenate((sub_B, np.zeros((dim**2,dim**2))), axis=1)
    B2=np.concatenate(( np.zeros((dim**2,dim**2)), sub_B),axis=1)
    B=np.concatenate((B1,B2), axis=0)# a changer construction matrice B
    return B

def bord_M_aleatoire(dimension, proba):
    M=np.zeros((dimension**2,dimension**2))
    for i in range(dimension**2):
        for j in range(dimension**2):
            if j % 10==0 or j % 10==9:
                M[i,j]=np.random.binomial(1, proba)
            elif j<=9 or j>=90:
                M[i,j]=np.random.binomial(1, proba)
    return M       
        
    
def cov_to_cor(B):
    inv_diag_B=np.linalg.inv(sqrtm(np.diag(np.diag(B))))
    inv_diag_B=np.copy(inv_diag_B.real)
    cor_B=np.dot(inv_diag_B,np.dot(B,inv_diag_B))
    return cor_B


########################################################################
    #covariance 1d
def Balgovind_1D(dim,L):
    B=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            r=abs(i-j)*1.
            B[i,j]=(1+r/L)*(math.exp(-r/L))

    return B

def expontielle_1D(dim,L):
    B=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            r=abs(i-j)*1.
            B[i,j]=math.exp(-r/L)

    return B

def Gaussian_1D(dim,L):
    B=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            r=abs(i-j)*1.
            B[i,j]=math.exp(-r**2/(2*L**2))

    return B
    
#Blgovind
def Balgovind_noniso(dim,L,rayon): #diag_vect: vector of dim, variance of each point
    sub_B=np.zeros((dim**2,dim**2))
    for i in range(dim**2):
        (a1,b1)=get_index_2d(dim,i)
        for j in range(dim**2):
            (a2,b2)=get_index_2d(dim,j) #reprends les donnees caracterisennes
            r=math.sqrt((a1-a2)**2+(b1-b2)**2)
            sub_B[i,j]=(1+r/L)*(math.exp(-r/L))
            
            rr1 = math.sqrt((a1-5)**2 + (b1-5)**2) #cercle in the middle
            rr2 = math.sqrt((a2-5)**2 + (b2-5)**2)
            if rr1<=rayon: #inside the cercle
                sub_B[i,j]=sub_B[i,j]*4
            else:
                sub_B[i,j]=sub_B[i,j]*0.25
            if rr2<=rayon:
                sub_B[i,j]=sub_B[i,j]*4
            else:
                sub_B[i,j]=sub_B[i,j]*0.25
                
    B1=np.concatenate((sub_B, np.zeros((dim**2,dim**2))), axis=1)
    B2=np.concatenate(( np.zeros((dim**2,dim**2)), sub_B),axis=1)
    B=np.concatenate((B1,B2), axis=0)# a changer construction matrice B
    return B    

def block_R_50(vect_variance): #num_elt in each block
    count = 0
    R_list = []
    for j in range(5):
        list_current = []
        
        for i in range(5):
            list_current.append(np.ones((10, 10))*vect_variance[count])
            count += 1
        
        R_list.append(list_current)
    
    return np.block(R_list)
    
    

if __name__ == "__main__":
    im = plt.imshow(0.001*Balgovind(10,2)[:100,:100])
    plt.colorbar(im)
    plt.show()

    im = plt.imshow(0.001*Balgovind_1D(100,10))
    plt.colorbar(im)
    plt.show()
    
    im = plt.imshow(Balgovind_general(300,300,20))
    plt.colorbar(im)
    plt.show()
    
    D = np.random.lognormal(0, 1, 25)
    #plt.savefig("Figures/R_bal_3.eps", format ='eps')
    
    H_1D_uv = np.zeros((2500,10000))

    for i in range(50):
        for j in range(50):
            
            H_1D_uv[i*50+j,2*i*100+2*j] = 1
            
            H_1D_uv[i*50+j,(2*i+1)*100+2*j] = 1
            
            H_1D_uv[i*50+j,2*i*100+2*j+1] = 1
            
            H_1D_uv[i*50+j,(2*i+1)*100+2*j+1] = 1
            
    H_1D_uv = H_1D_uv/4.
    
    H_1D_33_obs = np.zeros((1089,10000))
    
    for i in range(33):
        for j in range(33):
            
            H_1D_33_obs[i*33+j,3*i*100+3*j] = 1
            
            H_1D_33_obs[i*33+j,(3*i+1)*100+3*j] = 1
            
            H_1D_33_obs[i*33+j,(3*i+2)*100+3*j] = 1
            
            H_1D_33_obs[i*33+j,3*i*100+3*j+1] = 1
            
            H_1D_33_obs[i*33+j,(3*i+1)*100+3*j+1] = 1
            
            H_1D_33_obs[i*33+j,(3*i+2)*100+3*j+1] = 1
            
            H_1D_33_obs[i*33+j,3*i*100+3*j+2] = 1
            
            H_1D_33_obs[i*33+j,(3*i+1)*100+3*j+2] = 1
            
            H_1D_33_obs[i*33+j,(3*i+2)*100+3*j+2] = 1            
            
    H_1D_33_obs = H_1D_33_obs/9.
    
    
    H_1D_55_obs = np.zeros((400,10000))
    for i in range(20):
        for j in range(20):
            
            H_1D_55_obs[i*20+j,5*i*100+5*j] = 1
            
            H_1D_55_obs[i*20+j,(5*i+1)*100+5*j] = 1
            
            H_1D_55_obs[i*20+j,(5*i+2)*100+5*j] = 1

            H_1D_55_obs[i*20+j,(5*i+3)*100+5*j] = 1
            
            H_1D_55_obs[i*20+j,(5*i+4)*100+5*j] = 1

            H_1D_55_obs[i*20+j,5*i*100+5*j+1] = 1
            
            H_1D_55_obs[i*20+j,(5*i+1)*100+5*j+1] = 1
            
            H_1D_55_obs[i*20+j,(5*i+2)*100+5*j+1] = 1

            H_1D_55_obs[i*20+j,(5*i+3)*100+5*j+1] = 1
            
            H_1D_55_obs[i*20+j,(5*i+4)*100+5*j+1] = 1

            H_1D_55_obs[i*20+j,5*i*100+5*j+2] = 1
            
            H_1D_55_obs[i*20+j,(5*i+1)*100+5*j+2] = 1
            
            H_1D_55_obs[i*20+j,(5*i+2)*100+5*j+2] = 1

            H_1D_55_obs[i*20+j,(5*i+3)*100+5*j+2] = 1
            
            H_1D_55_obs[i*20+j,(5*i+4)*100+5*j+2] = 1
            
            H_1D_55_obs[i*20+j,5*i*100+5*j+3] = 1
            
            H_1D_55_obs[i*20+j,(5*i+1)*100+5*j+3] = 1
            
            H_1D_55_obs[i*20+j,(5*i+2)*100+5*j+3] = 1

            H_1D_55_obs[i*20+j,(5*i+3)*100+5*j+3] = 1
            
            H_1D_55_obs[i*20+j,(5*i+4)*100+5*j+3] = 1          

            H_1D_55_obs[i*20+j,5*i*100+5*j+4] = 1
            
            H_1D_55_obs[i*20+j,(5*i+1)*100+5*j+4] = 1
            
            H_1D_55_obs[i*20+j,(5*i+2)*100+5*j+4] = 1

            H_1D_55_obs[i*20+j,(5*i+3)*100+5*j+4] = 1
            
            H_1D_55_obs[i*20+j,(5*i+4)*100+5*j+4] = 1                
    H_1D_33_obs = H_1D_33_obs/25.
            
    
