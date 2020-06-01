import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import FuzzyCMeans as fm

def sample_fcm_image():
    
    #Loading image
    img = cv2.imread('3096.jpg')
    img = img_as_float(img)
    x = np.reshape(img,(img.shape[0]*img.shape[1],3),order='F')
    
    #Applying FCM to pixels
    cluster_n = 2
    expo = 2 
    min_err = 0.001 
    max_iter = 500 
    verbose = 0
    m,c = fm.fcm(x,cluster_n,expo,min_err,max_iter,verbose)
    m = np.reshape(m,(img.shape[0],img.shape[1]),order='F')
    
    #Replace pixel intensity with centers found by FCM or replace pixel intensity with median for each cluster
    simg = fm.keep_center(img,m,c,verbose)
    # simg = fm.calc_median(img,m,verbose)
    
    #Preview output image
    plt.imshow(simg[:,:,::-1])
    plt.show()
    
def sample_fcm_data():
    
    #Generate random data
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,random_state=0)
    X = StandardScaler().fit_transform(X)
    
    #Applying FCM to data
    cluster_n = 3
    expo = 2 
    min_err = 0.001 
    max_iter = 500 
    verbose = 0
    m,c = fm.fcm(X,cluster_n,expo,min_err,max_iter,verbose)
    
    #Plotting the results obtained
    colr = ['red','green','blue']
    fig, (ax0, ax1) = plt.subplots(ncols=2, constrained_layout=True)
    for i in range(3):
        indx = np.where((labels_true==i))
        ax0.scatter(X[indx,0],X[indx,1],color=colr[i])
        ax0.set_xlabel("X[0]")
        ax0.set_ylabel("X[1]")
        ax0.set_title('Original')
    for i in range(3):
        indx = np.where((m==i))
        ax1.scatter(X[indx,0],X[indx,1],color=colr[i])
        ax1.set_xlabel("X[0]")
        ax1.set_ylabel("X[1]")
        ax1.set_title('Using FCM')
    plt.show()  
    
if __name__ == "__main__":
    sample_image = 0
    sample_data  = 0
    if sample_image:
        sample_fcm_image()
    if sample_data:
        sample_fcm_data()