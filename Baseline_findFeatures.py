import cv2
import numpy as np
from scipy.cluster.vq import *
from tqdm import tqdm
import time
from sklearn import preprocessing


def train_findFeatures(images,numWords = 1000):
    des_list = []
    total_key_points = 0
    print("\nTraining: Applying SIFT ...")
    for image in tqdm(images):
        kp1, des = cv2.xfeatures2d.SURF_create().detectAndCompute(image, None);
        des_list.append(des)
        total_key_points = total_key_points + len(kp1)

    # Stack all the descriptors vertically in a numpy array
    # descriptors = des_list[0]
    # for descriptor in tqdm(des_list[1:]):
    #     descriptors = np.vstack((descriptors, descriptor))

    print("\nTraining: Stack all the descriptors...")
    my_descriptors = np.zeros([total_key_points,64])
    base_index = 0
    for descriptor in tqdm(des_list):
        length = len(descriptor)
        my_descriptors[base_index:base_index+length] = descriptor
        base_index = base_index + length

    # print("My_descriptors", type(my_descriptors))
    # print("==========")
    # print("My_descriptors",my_descriptors.shape)
    # print("==========")

    descriptors = my_descriptors

    # Perform k-means clustering
    localtime = time.asctime(time.localtime(time.time()))
    print("\ncurrent time :", localtime)

    print("\nTraining: Start k-means: %d words, %d key points" % (numWords, descriptors.shape[0]))
    kmstart = time.clock()
    voc, variance = kmeans(descriptors, numWords, 1, thresh = 1e-2)
    print("Time for k means",time.clock()-kmstart,"s")
    # Calculate the histogram of features
    im_features = np.zeros((len(images), numWords), "float32")

    print("\nTraining: Applying BOW...")
    for i in tqdm(range(len(images))):
        words, distance = vq(des_list[i],voc)
        for w in words:
            im_features[i][w] += 1

    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)

    idf = np.array(np.log((1.0 * len(images) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    a = 1
    im_features = im_features * idf * a
    # im_features = preprocessing.normalize(im_features, norm='l2')

    return (im_features,voc,idf)


def test_findFeatures(images,voc, idf, numWords = 1000):
    des_list = []
    print("\nTesting: Applying SIFT ...")
    for image in tqdm(images):
        kp1, des = cv2.xfeatures2d.SURF_create().detectAndCompute(image, None);
        des_list.append(des)


    # Calculate the histogram of features
    im_features = np.zeros((len(images), numWords), "float32")

    print("\nTesting: Applying BOW...")
    for i in tqdm(range(len(images))):
        words, distance = vq(des_list[i],voc)
        for w in words:
            im_features[i][w] += 1

    test_features = im_features * idf
    # test_features = preprocessing.normalize(test_features, norm='l2')
    return test_features