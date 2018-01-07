from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import pickle
import cv2
import glob
import time
import os

from feature_extraction import extractFeatures, readImages, configParams

def extractFeaturesImages(imgs):
    # Create a list to append feature vectors to
    imagesFeatures = []
    # Iterate through the list of images
    for img in range(0, len(imgs)):
        # Read in each one by one        
        image = imgs[img]
        
        hog_feat = configParams['use_hog_feat']
        spatial_feat = configParams['use_spatial_feat']
        hist_feat = configParams['use_hist_feat']

        imgFeatures = extractFeatures(image, verbose=False, hog_feat=hog_feat, spatial_feat=spatial_feat, hist_feat=hist_feat)

        imagesFeatures.append(imgFeatures)
    # Return list of feature vectors
    return imagesFeatures 

def fitSvm(X, labels, verbose):
    
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, labels, test_size=0.2,random_state=rand_state) 
    
    # Use a linear SVC 
    svc = LinearSVC()
    
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    if verbose:
        print("\n",round(t2-t, 2), 'Seconds to train SVC...')
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        
        t=time.time()    
        n_predict = 10
        print('My SVC predicts:     ', svc.predict(X_test[0:n_predict]))
        print('For these',n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(" ",round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    
    return svc, X_scaler 


svc, X_scaler = None, None


if __name__ == '__main__':

    vehicleImages = readImages('./data/vehicles')
    nonVehicleImages = readImages('./data/non-vehicles')

    vehiclesFeatures = extractFeaturesImages(vehicleImages)
    nonVehiclesFeatures = extractFeaturesImages(nonVehicleImages)

    X = np.vstack((vehiclesFeatures, nonVehiclesFeatures)).astype(np.float64)  

    # Define the labels vector
    labels = np.hstack((np.ones(len(vehiclesFeatures)), np.zeros(len(nonVehiclesFeatures))))

    svc, X_scaler = fitSvm(X, labels, verbose=True)
    pickle.dump([svc, X_scaler], open( "./classifier_pickle.p", "wb" ) ) 