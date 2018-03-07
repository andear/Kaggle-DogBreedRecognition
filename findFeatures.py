import cv2
import numpy as np
from scipy.cluster.vq import *
from tqdm import tqdm

def train_findFeatures(images):

    numWords = 1000

    des_list = []
    print("\nTraining: Applying SIFT ...")
    for image in tqdm(images):
        kp1, des = cv2.xfeatures2d.SURF_create().detectAndCompute(image, None);
        des_list.append(des)

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0]
    print("\nTraing: Stack all the descriptors...")
    for descriptor in tqdm(des_list[1:]):
        descriptors = np.vstack((descriptors, descriptor))

    # print("descriptors", type(descriptors))
    # print("==========")
    # print(descriptors.shape)
    # print("==========")
    # print(descriptors)

    # Perform k-means clustering
    print ("\nTraining: Start k-means: %d words, %d key points" %(numWords, descriptors.shape[0]))
    voc, variance = kmeans(descriptors, numWords, 1)

    # Calculate the histogram of features
    im_features = np.zeros((len(images), numWords), "float32")

    print("\nTraining: Applying BOW...")
    for i in tqdm(range(len(images))):
        words, distance = vq(des_list[i],voc)
        for w in words:
            im_features[i][w] += 1
    return (im_features,voc)

def test_findFeatures(images,voc):
    numWords = 1000
    des_list = []
    print("\nTesting: Applying SIFT ...")
    for image in tqdm(images):
        kp1, des = cv2.xfeatures2d.SURF_create().detectAndCompute(image, None);
        des_list.append(des)

    # Stack all the descriptors vertically in a numpy array
    descriptors = des_list[0]
    print("\nTesting: Stack all the descriptors...")
    for descriptor in tqdm(des_list[1:]):
        descriptors = np.vstack((descriptors, descriptor))

    # Calculate the histogram of features
    im_features = np.zeros((len(images), numWords), "float32")

    print("\nTesting: Applying BOW...")
    for i in tqdm(range(len(images))):
        words, distance = vq(des_list[i],voc)
        for w in words:
            im_features[i][w] += 1
    return im_features