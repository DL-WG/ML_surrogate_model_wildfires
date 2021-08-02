# -*- coding: utf-8 -*-
# assimilation shallow water
import numpy as np
from scipy import optimize
import scipy
from scipy.sparse import csr_matrix

def VAR_3D(xb,Y,H,B,R): #booleen=0 garde la trace
    dim_x = xb.size
    #dim_y = Y.size
    Y.shape = (Y.size,1)
    xb1=np.copy(xb)
    xb1.shape=(xb1.size,1)
    K=np.dot(B,np.dot(np.transpose(H),np.linalg.pinv(np.dot(H,np.dot(B,np.transpose(H)))+R))) #matrice de gain
    
    A=np.dot(np.dot((np.eye(dim_x)-np.dot(K,H)),B),np.transpose((np.eye(dim_x)-np.dot(K,H))))+np.dot(np.dot(K,R),np.transpose(K))
    vect=np.dot(H,xb1)
    xa=np.copy(xb1+np.dot(K,(Y-vect)))

    
    return xa,A


def DI01(xb,Y,H,B,R): #booleen=1 garde la trace
    (dimension,_)=B.shape
    dim_y=Y.size

    xb1=np.copy(xb)
    xb1.shape=(xb1.size,1)
    K=np.dot(B,np.dot(np.transpose(H),np.linalg.pinv(np.dot(H,np.dot(B,np.transpose(H)))+R))) #matrice de gain
    A=np.dot(np.dot((np.eye(dimension)-np.dot(K,H)),B),np.transpose((np.eye(dimension)-np.dot(K,H))))+np.dot(np.dot(K,R),np.transpose(K))
    vect=np.dot(H,xb1)
    xa=np.copy(xb1+np.dot(K,(Y-vect)))
    B_next=np.copy(A)
        
    Jb=np.dot(np.dot(np.transpose(xa-xb1),(np.linalg.pinv(B))),(xa-xb1))
    Jb=Jb[0][0]
    Jr=np.dot(np.dot(np.transpose(Y-np.dot(H,xa)),(np.linalg.pinv(R))),(Y-np.dot(H,xa)))
    Jr=Jr[0][0]

    sb=2*Jb/np.trace(np.dot(K,H))
    sr=2*Jr/np.trace(np.eye(dim_y)-np.dot(H,K))

    return xa,A,sb*B,sr*R,sb,sr
    
#sparse_H_1D_33 = scipy.sparse.load_npz('data_POD/H_1D_33_sparse.npz')
#
#def field_to_temperature(xb): #booleen=0 garde la trace, #xb:pod mode
#    
#    H = sparse_H_1D_33.todense()
#    
#    u_pod = np.loadtxt("data_POD/u_pod_"+str(9)+"sim_new_100.txt")
#    
#    field = np.dot(u_pod[:,:100], xb.reshape(xb.size,1))
#    
#    field.shape = (300,300)
#    
#    temperature_gloabl = np.zeros((300,300))
#    
#    temperature_gloabl[field >= 4] = 50
#    
#    temperature_gloabl[field == 3] = 800
#    
##    temperature_gloabl[field_moins_1 == 3 ] = 400
##    
##    temperature_gloabl[field_moins_2 == 3 ] = 200
##    
##    temperature_gloabl[field_moins_3 == 3 ] = 150
#    
#    temperature_observed = np.dot(H,
#                                  temperature_gloabl.reshape(temperature_gloabl.size,1))   
#    
#    temperature_observed.shape = (100 ,100)
#      
#    return temperature_observed


def parameter_3dvar(x,xb,Y,H,B,R):#with pod for xb
    
    B_inv = np.linalg.pinv(B)
    
    R_inv = scipy.sparse.linalg.inv(R)
    
    Jb = np.dot(np.dot((x-xb).reshape(1,xb.size),B_inv),(x-xb).reshape(xb.size,1))
    
    Jo = np.dot(np.dot((Y-field_to_temperature(x).ravel()).reshape(1,Y.size),R_inv.todense()),
                np.dot((Y-field_to_temperature(x).ravel()).reshape(Y.size,1)))
    
    return Jb+Jo

#################################################################
# test
    
def opt_func(x):

    return parameter_3dvar(x,xb,Y,H,B,R)



#################################################################
# test
    
#def func(z):
#    x, y = z
#    return x**2*y**2 + x**4 + 1
#
#sol1 = optimize.minimize(func, [1,100], tol=1e-10,
#                         method='l-bfgs-b')
#################################################################
def x_to_y(X): # averaging in 2*2 windows (4 pixels)
    dim = int(X.shape[0])
    Y = np.zeros((int(dim/2),int(dim/2)))
    print (Y.shape)
    for i in range(int(dim/2)):
        for j in range(int(dim/2)):
            Y[i,j] = X[2*i,2*j] + X[2*i+1,2*j] + X[2*i,2*j+1] + X[2*i+1,2*j+1]
            Y[i,j] = Y[i,j]/4. 
            
#            Y_noise = np.random.multivariate_normal(np.zeros(100),0.0000 * np.eye(100))
#            Y_noise.shape = (10,10)
#            Y = Y + Y_noise
    return Y
    

#H_1D_uv = np.zeros((100,400))

#def H_1D(X, ratio):
#    dim_y = int((X.shape[0]*1.)/ratio)
#    
#    H_1D_uv = np.zeros((dim_y**2,4*dim_y**2))
#    
#    print(H_1D_uv.shape)
#    
#    for i in range(X.shape[0]):
#        for j in range(X.shape[0]):
#            
#            print(i*dim_y+j,2*i*2*dim_y+2*j)
#            print (i, dim_y, j)
#            print("33333333333333333333333333333333333333333333")
#            H_1D_uv[i*dim_y+j,2*i*2*dim_y+2*j] = 1
#            
#            #print(i*dim_y+j,(2*i+1)*2*dim_y+2*j)
#            H_1D_uv[i*dim_y+j,(2*i+1)*2*dim_y+2*j] = 1
#            
#            H_1D_uv[i*dim_y+j,2*i*2*dim_y+2*j+1] = 1
#            
#            H_1D_uv[i*dim_y+j,(2*i+1)*2*dim_y+2*j+1] = 1
#            
#    return H_1D_uv

#H_1strow = np.concatenate((H_1D_uv,np.zeros((100,400))),axis = 1)
#
#H_2ndrow = np.concatenate((np.zeros((100,400)),H_1D_uv),axis = 1)
#
#
#H = np.concatenate((H_1strow,H_2ndrow),axis = 0)


#H_1D_uv = np.zeros((2500,10000))
#
#for i in range(50):
#    for j in range(50):
#        
#        H_1D_uv[i*50+j,2*i*100+2*j] = 1
#        
#        H_1D_uv[i*50+j,(2*i+1)*100+2*j] = 1
#        
#        H_1D_uv[i*50+j,2*i*100+2*j+1] = 1
#        
#        H_1D_uv[i*50+j,(2*i+1)*100+2*j+1] = 1
#        
#H_1D_uv = H_1D_uv/4.

#H_1strow = np.concatenate((H_1D_uv,np.zeros((2500,10000))),axis = 1)
#
#H_2ndrow = np.concatenate((np.zeros((2500,10000)),H_1D_uv),axis = 1)
#
#
#H = np.concatenate((H_1strow,H_2ndrow),axis = 0)


if __name__ == "__main__":
    
    u_pod = np.loadtxt("data_POD/u_pod_"+str(9)+"sim_new_100.txt")
    
    field = np.loadtxt('result_MC/LA_MC_'+str(0)+str(0)+'.txt')
    
    xb = np.dot (u_pod[:,:100].T,field.reshape(field.size,1))

    ss = field_to_temperature(xb)
    
    
    B = np.eye(100)
    
    R = csr_matrix(np.eye(10000))
    
    H = sparse_H_1D_33.todense()
    
    Y = np.ones((100,100))
    
    
    def parameter_3dvar(x,xb,Y,H,B,R):#with pod for xb
        
        xb = np.array(xb)
        
        x.shape = (x.size,1)
        
        xb.shape = (xb.size,1)
        
        Y.shape = (Y.size,1)
        
        B_inv = np.linalg.pinv(B)
        
        R_inv = scipy.sparse.linalg.inv(R)
        
        print('11111111111111111111111111111111111',(x-xb).shape)
        
        Jb = np.dot(np.dot((x-xb).reshape(1,xb.size),B_inv),(x-xb).reshape(xb.size,1))
        
        Jo = np.dot(np.dot((Y.ravel()-field_to_temperature(x).ravel()).reshape(1,Y.size),R_inv.todense()),
                    (Y.ravel()-field_to_temperature(x).ravel()).reshape(Y.size,1))
        
        return Jb+Jo
    
    #################################################################
    # test
        
    def opt_func(x):
    
        return parameter_3dvar(x,xb,Y,H,B,R)
    
    sol1 = optimize.minimize(opt_func, list(xb.ravel()), tol=1e-10, method='l-bfgs-b')
    
    
