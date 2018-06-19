# Self-Driving Car Engineer Nanodegree Program
## Vehicle-Detection Project

The goals:

* Use computer vision techniques to detect the cars in the image or video

The steps of this project are the following: 

* Find the model to classify the car image and noncar image.
* Construct features set from the image data set.
* Use the feature set to train the model.
* Implement a sliding-window technique to search for cars with the trained model
* Creating a heatmap of recurring detections in subsequent frames of a video stream to reject outliers and follow detected vehicles.



[//]: # (Image References)
[image3]: ./output_images/03_feature_selection.png
[image4]: ./output_images/04_boxes_1.png
[image5]: ./output_images/05_boxes_2.png
[image6]: ./output_images/06_boxes_3.png
[image6a]: ./output_images/06a_boxes_4.png
[image7]: ./output_images/07_all_detections.png
[image8]: ./output_images/08_heatmap.png
[image9]: ./output_images/09_heatmap_threshold.png
[image10]: ./output_images/10_label_heatmap.png
[image11]: ./output_images/11_final_boxes.png
[image12]: ./output_images/12_all_test_detects.png
[test1]: ./output_images/test1.png
[test2]: ./output_images/test2.png
[test3]: ./output_images/test3.png
[test4]: ./output_images/test4.png
[test5]: ./output_images/test5.png
[test6]: ./output_images/test6.png
[debug1]: ./output_images/debug1.png
[debug2]: ./output_images/debug2.png
[video1]: ./test_video_out.mp4
[video2]: ./test_video_out_2.mp4
[video3]: ./project_video_out.mp4


### Files:
* main.py: the main code to start the pipeline 
* feature.py: this part of code defines some features class such as SpatialFeature, ColorHistFeature, HogFeature which are inherted from an ImageFeature base class. Each feature class contains the method to extract the featue, set the feature scaler(nomalizer) and so on. 
* vehicle.py: this core part of cone defines a Car class which record the car information and contains the methods to detect car.   
* video_helper.py: the code for processing the video


### Find good model/classifier 

#### 1. Extract features from image set

Thanks to the Udacity, we are provided with the image dateset which contains more than 8000 car images and 8000 noncar images. Each image is 64 x 64 pixels. If we use deep learning detect method, we can use all image pixels as the input. However, we use image feature not all image pixel here to train the model. 

There are several ways to extract the features, such as HOG(histogram of the oriented graident), color histogram and so on. We can use one of them or all of them to get the feature vector. For example, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. But be careful. don't forget to normalize your features. The extract_feature() function in the lesson is not for mutiple feature-extraction. Here, I implement some feature class in feature.py which has it own feature-extract function and normailized function. 

#### 2. Choose the feature and model

In order to get an accurate car/noncar classify result. We need to choose the model and features to use.
For models, we can choose naive byes model, decision tree model or svm model.
For features, we can choose Hog, binned color, histogram of color.
For both models and features, there are many parameter to set. So it is difficult to find the best model and feature on instinct. So I write a function called search_best_combination in the feature.py to help us to find the best ones, but it check the accuarcy and train time one by one which is time-consuming. The result are recorded into csv using pandas. Some of the search result is shown below:

![alt text][image3]


#### 3. Train the model
Once we choose the best model and features. We can train the model to make it has the strong ability to classify car and noncar. Before training the model, don't forget to normalize the features and shuffle the train and test data.
From many experiments, my combination use all features and SVM as classifier:
* Color space: 'RGB'
* Color_Spatial_Feature: True, spatial_size = (16, 16)
* Color_Hist_Feature: True, hist_bins = 16
* Hog_Feature: True, orient = 9, pix_per_cell = 16, cell_per_block=2, hog_channel=1. 

As result the model's accuracy is more than 99.3%


### Detect the car in the image/video

#### 1. Sliding window search

In the section titled "Method for Using Classifier to Detect Cars in an Image" I modified the method find_cars to detect method of Car class from the lesson materials. The modified method combines three feature extraction with a sliding window search, but rather than perform feature extraction on each window individually which can be time consuming, the HOG features are extracted for the entire image (or a selected portion of it) and then these full-image features are subsampled according to the size of the window and then fed to the classifier. The method performs the classifier prediction on the HOG features for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive ("car") prediction.

I explored several configurations of window sizes and positions, with various overlaps in the X and Y directions. The following four images show the configurations of all search windows in the final implementation, for small (1x), medium (1.5x, 2x), and large (3x) windows:

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image6a]

#### 2. Reject outliers
Because a true positive is typically accompanied by several positive detections, while false positives are typically accompanied by only one or two detections, a combined heatmap and threshold is used to differentiate the two. The add_heat function increments the pixel value (referred to as "heat") of an all-black image the size of the original image at the location of each detection rectangle. Areas encompassed by more overlapping rectangles are assigned higher levels of heat. The following image is the resulting heatmap from the detections in the image above:

![alt text][image8]

A threshold is applied to the heatmap (in this example, with a value of 1), setting all pixels that don't exceed the threshold to zero. The result is below:

![alt text][image9]

The `scipy.ndimage.measurements.label()` function collects spatially contiguous areas of the heatmap and assigns each a label:

![alt text][image10]

And the final detection area is set to the extremities of each identified label:

![alt text][image11]

For the video, we can integrated and thresholded car boxes in recent 5 frames.


#### 3. Show the detect result

The results of passing all of the project test images through the above pipeline are displayed in the images below:

![alt text][test1]
![alt text][test2]
![alt text][test3]
![alt text][test4]
![alt text][test5]
![alt text][test6]

By the way, the debug mode of the pipeline also be implemented. Some important information about the selected features and classifer and the heatmap are annotated in the image.

![alt text][debug1]
![alt text][debug2]


### Pipeline (video)

#### 1.Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's the project video [link to my video result](./project_video_out.mp4)
Here's the debug project video [link to my video result](./project_video_debug_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems that I faced while implementing this project were mainly concerned with detection accuracy. It can not reach a very high accuracy with only one features. I attempt to use two or three features to get a higher accuracy, but I failed in the begining. Because features need to be normalized repsectively. Each feature need its own normailizer(StandScaler), so that the weight of each feature is similar. Finally, after I  modifying the code, each feature can be normalized respectively, and the accuarcy can reach up to more than 99.3%

In order to find the position of cars in the image, we need to slide many windows pf different size. The trained classifier needs to decide each window is car box or not. It is time-consuming. I should find another way to help the pipeline to find the car position easily or reduce the car box.


###Acknowledge

* Udacity
* https://github.com/jeremy-shannon
