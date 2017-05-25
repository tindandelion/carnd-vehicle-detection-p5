# Vehicle Detection Project

## Project goals

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a
  labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color
  features, as well as histograms of color, to your HOG feature vector;
* Implement a sliding-window technique and use your trained classifier to search
  for vehicles in images;
* Run your pipeline on a video stream and create a heat map of recurring
  detections frame by frame to reject outliers and follow detected vehicles;
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## Project files

The submission includes the following files:

* `feature-selection.ipynb` is a Jupyter notebook that contains code to select
  feature extraction parameters and train a classifier;
* `vehicle-delection.ipynb` is a Jupyter notebook where I implement the vehicle
  searching algorithm and apply it to the project's video file;
* `common.py` is a Python file that contains code shared between notebooks;
* `test_images/project_video.mp4` is a video output produced by the pipeline;
* `README.md` is a report that summarizes the results.

## Solution description


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features
    from the training images.

The code for this step is contained in the first code cell of the IPython
notebook (or in lines # through # of the file called `some_file.py`).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an
example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters
(`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random
images from each of the two classes and displayed them to get a feel for what
the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of
`orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

## Parameter selection

The number of feature selection parameters to be tuned is quite large and it's
not obvious how they would impact the performance of the detection algorithm. I
decided to try various combinations and select those where the classifier would
show the best accuracy. 

To accomplish this task, I decided to use the grid search approach, implemented
in `sklearn`. As the exhaustive search through all parameter combinations would
take a lot of time, I ended up using a randomized grid search with a limited
number of iterations that would try out different combinations selected at
random. 

`sklearn` library provides the following classes that fit the task at hand: 

* `Pipeline` class allows to combine different transformations and classifiers
  into a tunable pipeline;
* `RandomizedGridSearch` class implements the search algorithm that tries out
  different parameter combinations and keeps track of the best results. 
  
In order to plug the feature extration algorithm into a pipeline, I implemented
a class `FeatureExtractor` that is essentially an adapter around
`extract_features()` function, suitable for prugging into the grid search. 

The complete classification pipeline consists of the following blocks:

* `FeatureExtractor` that extracts HOG, spacial, and color features from the raw
  images;
* `StandardScaler` scales the features to zero mean and unit variance;
* `LinearSVC` is a Support Vector Machine classifier to do the job. 

The pipeline is then plugged into the `RandomizedGridSearch`. 

Here are a few parameter combinations that the grid search found to show the
best performance (extracted from `output_model/gridsearch_result.csv`):

At the end, I save the best trained classifier into a Pickle file, along with
the parameter values, for use in vehicle detection. Note that I remove the
`FeatureExtractor` instance from the pipeline before saving, as I decided to use
an optimized sliding window approach for vehicle detection (see below).



###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

