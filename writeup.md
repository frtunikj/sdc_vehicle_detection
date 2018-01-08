# **Vehicle Detection and Tracking**

[//]: # (Image References)
[image0]: ./output_images/training_images.png
[image1]: ./output_images/HOG_non_vehicle.png
[image2]: ./output_images/HOG_vehicle.png
[image3]: ./output_images/slidingwindow1.png
[image4]: ./output_images/slidingwindow2.png
[image5]: ./output_images/slidingwindow3.png
[image6]: ./output_images/hogsubsampling1.png
[image7]: ./output_images/hogsubsampling2.png
[image8]: ./output_images/hogsubsampling3.png
[image9]: ./output_images/heatmap1.png
[image10]: ./output_images/heatmap4.png
[image11]: ./output_images/heatmap3.png

### Goals of the project

The goals / steps of this project were the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Repository Files Description

This repository includes the following files:

* [01_feature_extraction.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/01_feature_extraction.ipynb) - performs a Histogram of Oriented Gradients (HOG) feature and color extraction on images.
* [02_classifier.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/02_classifier.ipynb) - loads and preprocesses training and testing labeled data, trains and evaluates the linear SVM classifier. 
* [03_sliding_window.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/03_sliding_window.ipynb) - implements a sliding-window algorithm which is used by the trained classifier to search for vehicles in images, draws bounding boxes for detected vehicles.
* [04_heatmap.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/04_heatmap.ipynb) - creates a heatmap of recurring detections frame/image by frame/image to reject outliers and follow detected images, draws estimated bounding boxes for detected vehicles.
* [05_video_vehicle_detection.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/05_video_vehicle_detection.ipynb) - outputs a video that contains the detected vehicles.

NOTE: The labeled training data can be found here: [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip).  These images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. In addition, it is possible to use the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) training data.  

Some example images for testing the pipeline on single frames are located in the [test_images](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/test_images) folder. The video called [project_video.mp4](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/project_video.mp4) is the video on which the pipeline should work well on.  

---

### Feature extraction

#### 1. Histogram of Oriented Gradients (HOG) and color feature extraction

First the training vehicle and non-vehicle images were explored (see second and third code cell in [01_feature_extraction.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/01_feature_extraction.ipynb)). There are 8792 vehicle images and 8968 non-vehicle images. Here is an example of those images:

![alt text][image0]

To differentiate between classes of objects vehicles vs non-vehicles structurals cues like shape are used normally. Gradients/derivative of specific directions captures some notion of shape and to allow for some variability in shape, a Histogram of Oriented Gradients (HOG) is commonly used. The idea of HOG is instead of using each individual gradient direction of each individual pixel of an image, to group the pixels into small cells of n x n pixels. For each cell, the gradient directions are computed and grouped into a number of orientation bins. Then the gradient magnitude in each sample is summed up. Stronger gradients contribute more weight to their bins, and in this way effects of small random orientations due to noise is reduced. At the end, the histogram provides the information regarding the dominant orientation of that cell. Doing this for all cells gives the final representation of the structure of the image. The HOG features keep the representation of an object distinct but also allows for some variations in shape. HOG feature extraction (based on Udacity code) is implemented in the function extract_hog_features() in Step 3 of [01_feature_extraction.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/01_feature_extraction.ipynb). 

In addition, color feature extraction is used to capture object appearance. The color features consisted of a downsampled image and histograms of intensity values in the individual channels of the image (see extract_color_features() function in Step 3 of [01_feature_extraction.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/01_feature_extraction.ipynb)). The extract_color_features() (based on Udacity code) function transforms the input image to the YCrCb color space.

A function that combines the HOG and color feature extraction is called extractFeatures() and can be found in Step 4 of [01_feature_extraction.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/01_feature_extraction.ipynb). 

HOG and color features were extracted from images of vehicle and non-vehicle classes. One example of each class is shown below.

![alt text][image1]

![alt text][image2]

#### 2. HOG and color feature extraction parameters

The selection of the parameters of the feature extraction (HOG or/and color) was driven by the measurement of the test accuracy for the liner SVM classifier (for details see next section "Training an SVM classifier"). Different parameter settings for HOG feature extraction and color feature extraction were explored. At the end the best accuracy of 98.54% was achieved.

The aforementioned accuracy was achieved with the following HOG parameters:

```
'use_gray_img':'False',
'hog_channel':'ALL',
'hog_cspace':'YCrCb',
'hog_n_orientations': 9,
'hog_pixels_per_cell': 8,
'hog_cells_per_block': 2,
```

In addition to the HOG feature extraction color feature extraction was used as mentioned earlier. The following parameter set for extracting color features were used at the end:

```
'color_cspace':'YCrCb',
'color_spatial_size':(32, 32),
'color_hist_bins':32,
'color_hist_range':(0, 256),
```
    
#### 3. Training a SVM classifier

The data set used for training the classifier that distinguish between vehicles and non-vehicles used 8792 vehicle images and 8968 non-vehicle images. As already mentioned, these images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. 80% of the data set was used to train the classifier and the remaining 20% was used to determine the accuracy of the classifier. To randomize the splitting of the data the built-in train_test_split function from sklearn was used and fed with random number between one and one hundred (see Step 2 and 3 of [02_classifier.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/02_classifier.ipynb)). Before training the classifier, the columns of the stacked feature vectors are normalized using StandardScaler() in order to avoid any particular feature dominating the others by sheer scale.

The classifier algorithm used is the Linear Support Vector Machine (SVM) as recommended by Udacity for this project. The SVM has advantages including being effective in high dimensional spaces even when the number of dimensions is almost as large or even larger than the number of sample such as this case. It is also said to be memory efficient and versatile (http://scikit-learn.org/stable/modules/svm.html). For each training vehicle/non-vehicle image, the HOG- and color-features are concatenated to form a one single feature vector (see Step 2 of [02_classifier.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/02_classifier.ipynb)). These feature vectors are stacked to form the matrix X, and labels car ("1") or non-car ("0") corresponding to each feature vector are stored in a vector of label's (see Step 2 of [02_classifier.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/02_classifier.ipynb)). At the end the linear SVM classifier is fitted to the training data in the function fitSvm(X, labels, verbose) in step 3 of [02_classifier.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/02_classifier.ipynb). At the end the linear SVM with the default classifier parameters and using HOG and color features was able to achieve a test accuracy of 98.54%. The results of the training are stored in [classifier_pickle.p](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/classifier_pickle.p) and used in the further steps.

```
4.16 Seconds to train SVC...
Test Accuracy of SVC =  0.9854
My SVC predicts:      [ 0.  0.  1.  0.  1.  1.  1.  0.  1.  0.]
For these 10 labels:  [ 1.  0.  1.  0.  1.  1.  1.  0.  1.  0.]
0.0043 Seconds to predict 10 labels with SVC
```

NOTE: For the training of the classifier (and experimenting with influence of the feature extraction) an AWS GPU instance was used. The reason was that not enough memory was available on the Linux VM.  

### Sliding Window Search for vehicle Detection

One approach for detecting vehicles in an image is to get a subregion of an image and run a classifier in that region to see if that patch contains a vehicle. Initially a mechanism was implemented where a list of windows (x, y-coordinates) was generated for a region of interest in an image, given window size and overlap (see slide_window() function Step 1 of [03_sliding_window.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/03_sliding_window.ipynb)). This windows list was then filtered for windows containing cars using the function search_windows(). The search_windows() function extracted HOG and color features for each window and classified the window as either containing a vehicle or not (by using the SVM classifier). Both functions were provided by Udacity. This method was tested on the testing images (see Step 2 of [03_sliding_window.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/03_sliding_window.ipynb)) and some result examples are given below:

![alt text][image3]

![alt text][image4] 

![alt text][image5] 

This process is computationally very expensive/time consuming, since the HOG features were extracted separately for each window. To reduce this computationally complexity, the above mechanism was altered (suggested by Udacity) so that the HOG features are extracted only once for a region of interest in an image. Later during window classification, only the portion of the large HOG feature array inside that window is considered. This mechanism is implemented in the function findCars() in Step 3 of [03_sliding_window.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/03_sliding_window.ipynb). The following images shows the results of applying findCars() function on the same test images from above:

![alt text][image6]

![alt text][image7] 

![alt text][image8] 

---

### Vehicle Detection and Tracking Video 

#### 1. Video Pipeline Implementation

The resulting video can be found under [output_video.mp4](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/videos/output_video.mp4). The complete pipeline for generating the video via processing video images/frames can be found in Step 1 of [05_video_vehicle_detection.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/05_video_vehicle_detection.ipynb).

#### 2. Methods for improving vehicle detection and tracking (reducion of false positives, combining overlapping bounding boxes)

Both functions search_windows() and findCars() returns a list of "hot" windows i.e. windows which are classified as containing a vehicle or not. By itself, this list contains some misclassified windows, so-called false positives. These single random instances of misclassification can typically disappear from one frame to the next, if the feature extraction and the image classification performs well. Based on this hypothesis (provided by Udacity), positive detections over a certain number of consecutive frames/buffers are accumulated into a heatmap. If an area in the image is consistently detected over consecutive frames, it will accumulate a higher heat value. On the other hand, single random instances of misclassified detections will not accumulate as much heat. Thresholding the heatmap should then remove these false positives. At the end the function label() is used to identify individual blobs in the heatmap. These blobs are assumed to correspond to a vehicle. Bounding boxes ar drawn around the blobs to indicate the vehicle.
This pipeline is implemented in [04_heatmap.ipynb](https://github.com/frtunikj/sdc_vehicle_detection/blob/master/04_heatmap.ipynb). The following images shows the results of applying get_heat_based_bboxes_one_image() and findCars() functions on the test images as above:

![alt text][image9]

![alt text][image10] 

![alt text][image11] 

After experimenting with the buffer length and threshold values the final values were set to:

```
'heat_threshold': 2,  
'buffer_len_hotwindows': 5,
```

---

### Discussion and potential for further improvements (undone work)

Three major difficulties (mainly w.r.t. finding the right parameters) were experienced in this project and are explained below:

* it was hard to estimate the position of the sliding window technique as well as the size of the window. A large window produced more false positives. 
* it was difficult to determine the length of buffer of consecutive frames/images (buffer_len_hotwindows) and the threshold of the heatmap (heat_threshold) in order to reject false positives but not reject true positives. 
* computational power of the Linux VM was not good for this project. However, switching to an AWS GPU instance helped a lot.

In order to improve the robustness of the pipeline one could try out the following things:

* use more than one scale (in findCars()) to find the windows and apply the heatmap on them
* use a CNN e.g. LeNet for the image classification problem.
