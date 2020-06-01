# Implementing Fuzzy C-Means in Python using numpy

Fuzzy C-Means(FCM) is a clustering algorithm which aims to partition the data into C homogeneous regions. `FuzzyCMeans.py` contains the implementation of FCM using numpy library only. Two demonstrations of the implemented FCM algorithm is provided in `sample.py`, which are as follows:
* In `sample_fcm_image()`, FCM is applied to an image. The output of applying FCM on image can be viewed in two ways:
	*  Uncomment the `simg = fm.keep_center(img,m,c,verbose)` line to replace each pixel intensity with the cluster center predicted by FCM algorithm. 
	*   Uncomment the `simg = fm.calc_median(img,m,verbose)` line to replace each pixel intensity with the median intensity of the pixels belonging to same cluster. 
*  In `sample_fcm_data()`, FCM is applied to synthetically generated data. The original labels of the data and the predicted labels using FCM are plotted beside each other to get a visual comparison about the performance of FCM algorithm. 

## Parameters for FCM
The following parameters can be tuned to modify the performance of FCM:
- `cluster_n`: The number of desired clusters.
- `expo`: The exponent value of FCM.
- `min_err`: Difference in the membership values of two consecutive iterations to declare convergence.
- `max_iter`: Maximum number of iterations of the FCM algorithm.
- `verbose`: To view the iterations during execution of FCM, set `verbose=1`.

By default, the seed value of numpy.random is set to 0 in the code. It can be changed if required. The function returns the predicted cluster id for each data point and cluster_n number of cluster centers.

## Executing the code
In order to execute the code, follow the steps below:
- [x] Clone the repo.  
- [x] `import FuzzyCMeans as fm` and directly use it in your code.
- [x] If you wish to see the demo, open `sample.py` and do the following:
	- Change `sample_image = 1` to apply FCM on image. An image from BSDS dataset is provided as example.
    - Change `sample_data  = 1` to apply FCM on synthetically generated data.

## Remarks

If you like this code or use it for coding, please inform by opening an issue. Also, if you want the derivation of FCM and explanation of code, you can mention it in issue. I will try to address it with a video explanation.
