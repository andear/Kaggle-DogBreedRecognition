# Kaggle-DogBreedRecognition
This is for the Dog Breed Recognition competition in kaggle(https://www.kaggle.com/c/dog-breed-identification), and all data set 
can be downlowad in previous link. 120 breeds in total. There are 10222 dog pictures in training set and 10331 pictures
in test set to be classified. All pictures are from ImageNet.(http://www.image-net.org/)

Here we use pretrained VGG and random forest classifier to achieve the accuracy of 82%, 
and this project is still in progress. 

## Baseline
Initially, we want to use SIFT, K-means and Bag of words to extract and select features. 

Then we use a classifier to classify these features. Since the data set is a little unbalanced, random forest with bagging is a good choice to reduce variance. 

This is considered as a baseline method.


However, baseline method does not work well in this really high dimensional data. I try to improve baseline method by applying TF-IDF to modify the feature vectors. it indeed increases the accuracy by about 1.5%(if k is chosen properly). But the overall accuracy is still under 12%.

## Problem with baseline method
we try different classifier like SVM, the situation is not getting better. So the bottleneck is how to find a way to extract informational features.

## Transfer Learning
Deep Neural Network show its "omnipotent" in many fields. However we are not considering Convolutional Neural Network at first because we don't have resources like GPU, and we cannot afford long time training.

Transfer Learning offers us the convience to use deep neural network without training it from scratch.

***to be completed


## prospect
### object proposal
To be continued

### fine tune