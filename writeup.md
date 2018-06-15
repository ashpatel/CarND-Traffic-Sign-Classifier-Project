# **Traffic Sign Recognition**

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[datadistro]: ./writeup_images/Data_distro.png "Visualization"
[newdatadistro]: ./writeup_images/New_data_distro.png "Visualization"
[webtestimages]: ./writeup_images/web_test_images.png "Test Images"
[trainingbefore]: ./writeup_images/training_before.png "Training Before"
[trainingafter]: ./writeup_images/training_after.png "Training After"
[probabilities]: ./writeup_images/probabilities.png "Probabilities"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used standard Python and Numpy to query the shape of the training data to ascertain the various analysis of the datasets.

* The initial size of training set is `34799`
* The size of the validation set is `4410`
* The size of test set is `12630`
* The shape of a traffic sign image is `32 x 32 x 3`
* The number of unique classes/labels in the data set is `43`

The file signnames.csv contains a mapping to the label IDs (0-42) to their corresponding names. This was read in and used to make the results more readable by using the names of the signs.

#### 2. Include an exploratory visualization of the dataset.

The following three bar graphs show the distribution of the labels (0-42) in the Training, Test and Validation data sets.  The distribution of data is very uneven, with many signs having about 1/10th of the samples as some of the other signs in the data sets.

![alt text][datadistro]

I additionally generated a preview of 6 random examples of each type of sign, so that I could understand the visual quality of the images. I noticed that many examples were blurry and badly lit.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Due to fact that many labels that a low number of images, I decided that I would augment the training data set by generating extra images for the signs that had too few samples (I picked 500 as the threshold).

For each sign type, I found all the images for that sign type and then I simply tripled the number of signs by copying the originals signs three times. Then with the new copies, I processed the images by using a random rotation (-10 to +10 degrees) and randonly brightening dark images (those images where the average pixel value was less than 75). This way the newly generated images would not be identical to the originals.

This generated about an additional 18000 new training images, bringing the total number of training images to 52,979

The training set distribution before and after can be seen in the following two visualizations.

![alt text][trainingbefore]    -----------> ![alt text][trainingafter]


Additionally I tried turning all the images to grayscale, but this didn't have as major impact on the accuracy, as augmenting the dataset with extra images, so the only other processing I did was normalized the RGB values between -1 to +1 (instead of 0 to 255).



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I started with the LeNet model from the previous exercise and modified it to deal with the shape of the images in the training data. I added an extra full layer (layer 4a), and I added multiple dropout layers to prevent overfitting.

With these changes I was reliably able to reach the target goal of 0.93 accuracy in 25 EPOCHS.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  valid padding, outputs 14x14x6 |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16    |
| RELU         | Activation |
|Max pooling  | 2x2 stride, valid padding, outputs 5x5x16   |
| Flatten   | input 5x5x16, output 400 |
| Fully connected		| input 400, output 300								|
| RELU         | Activation |
| Fully connected		| input 300, output 200								|
| RELU         | Activation |
| Dropout   | keep 75%   |
| Fully connected		| input 200, output 100								|
| RELU         | Activation |
| Dropout   | keep 75%   |
| Fully connected		| input 100, output 43	(Classification)							|
| Softmax				| with cross entropy, AdamOptimizer |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model was trained on a AWS GPU install (G2 Large).

The model was trained using the AdamOptimizer and a learning rate of 0.001. The model was trained in 25 Epochs with a batch size of 128.

As I was using a large AWS GPU instance with more memory, I experimented with increasing the batch size to 256 and 512, but this reduced my accuracy, so I kept the batch size at 128.

While I tried 30,40 and 50 Epochs, I settled on 25 as the additional iterations didn't yield any additional accuracy for the considerably longer training time.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.943
* test set accuracy of 0.931

I started with the Original LeNet architecture from the previous exercise.
I added an extra full layer so that the drop off from 400 to 43 could be more gradual. Additionally I was seeing very early overfitting with high early testing accuracy vs validation accuracy. I add two dropout layers to the last two fully connected layers. This helped get my accuracy over 0.93

### Test a Model on New Images

#### 1. Choose at least five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The follow are 10 additional German Traffic Signs that I found on the web.

![alt text][webtestimages]

I was unable to find images that were as badly lit as some of the images in the training data, but due to the resizing of the images to 32x32, there was heavy pixelation and loss of shapes in some of the signs that might make it hard for the model to work on some of the images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The new images were preprocessed in the same way as the training set (we only ended doing normalization) and then run thru the model to get the predicted signs.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Children Crossing    		| Children Crossing 	|
| Speed Limit (30km/h)  | Speed Limit (30km/h) 			|
| Road Work					| Road Work				|
| No Passing	      		| No Passing		|
| Beware Ice/Snow		| `Dangerous Curve To The Left`      							|
| Turn Left Ahead		| Turn Left Ahead      							|
| Right of Way at Next Intersection		| Right of Way at Next Intersection      							|
| Speed Limit (30km/h)		| Speed Limit (30km/h)      							|
| Vehicles over 3.5 tons prohibited		| Vehicles over 3.5 tons prohibited      							|
| Road Work		| Road Work      							|



The model was able to correctly guess 9 of the 10 traffic signs, which gives an accuracy of 90%. Very similar to the accuracy of the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The visualization of the top 5 softmax probabilities is presented below for all the additional test images.

![probabilities]

For most of the images the softmax probability was 1.0 and for the ones where it wasn't 100% sure the probability was 0.95 and higher. For the one where it predicted incorrectly (Beware Ice/Snow), it the highest prediction was 0.68, plus it had additional predictions for other labels.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
