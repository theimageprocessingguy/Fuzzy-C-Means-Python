import numpy as np

def calc_median(im,seg_img,verbose):
    n = len(np.unique(seg_img))
    f_img = np.zeros_like(im,dtype=np.int64)
    segs = list(np.unique(seg_img))
    for i in range(n):
        if verbose:
            print(i)
        mask_indx = np.where((seg_img==segs[i]))
        mask = np.zeros_like(im[:,:,0],dtype=np.int64)
        mask[mask_indx] = 1
        r = im[:,:,0]
        r_med = np.median(r[mask_indx])
        g = im[:,:,1]
        g_med = np.median(g[mask_indx])
        b = im[:,:,2]
        b_med = np.median(b[mask_indx])
        f_img[:,:,0] += mask * int(r_med * 256)
        f_img[:,:,1] += mask * int(g_med * 256)
        f_img[:,:,2] += mask * int(b_med * 256)
    return f_img

def keep_center(im,seg_img,center,verbose):
    n = len(np.unique(seg_img))
    f_img = np.zeros_like(im,dtype=np.int64)
    segs = list(np.unique(seg_img))
    for i in range(n):
        if verbose:
            print(i)
        mask_indx = np.where((seg_img==segs[i]))
        mask = np.zeros_like(im[:,:,0],dtype=np.int64)
        mask[mask_indx] = 1
        f_img[:,:,0] += mask * int(center[segs[i],0] * 256)
        f_img[:,:,1] += mask * int(center[segs[i],1] * 256)
        f_img[:,:,2] += mask * int(center[segs[i],2] * 256)
    return f_img

def init_memval(cluster_n, data_n):  
    U = np.random.random((cluster_n, data_n))
    val = sum(U)
    U = np.divide(U,np.dot(np.ones((cluster_n,1)),np.reshape(val,(1,data_n))))
    return U
    
def fcm(data,cluster_n,expo = 2,min_err = 0.001,max_iter = 500,verbose = 0):
    np.random.seed(0)
    U_old={}
    data_n = data.shape[0]
    U = init_memval(cluster_n, data_n)
    for i in range(max_iter):
        if verbose:
            print('Iteration: ',i)
        mf = np.power(U,expo)
        center = np.divide(np.dot(mf,data),(np.ones((data.shape[1], 1))*sum(mf.T)).T)
        diff = np.zeros((center.shape[0], data.shape[0]))
        if center.shape[1] > 1:
            for k in range(center.shape[0]):
                diff[k, :] = np.sqrt(sum(np.power(data-np.dot(np.ones((data.shape[0], 1)),np.reshape(center[k, :],(1,center.shape[1]))),2).T))
        else:	# for 1-D data
            for k in range(center.shape[0]):
                diff[k, :] = abs(center[k]-data).T
        dist=diff+0.0001;
        num = np.power(dist,(-2/(expo-1)))
        U = np.divide(num,np.dot(np.ones((cluster_n, 1)),np.reshape(sum(num),(1,num.shape[1])))+0.0001)
        U_old[i]=U;
        if i> 0:
            if abs(np.amax(U_old[i] - U_old[i-1])) < min_err:
                break
    U = np.argmax(U,axis=0)
    return U,center
