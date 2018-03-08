import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.vq import *
from findFeatures import train_findFeatures



# image_a = cv2.imread("data_gen/train/basset/4ff1a1c62acccfbb0bb1a6ecb4f29682.jpg",0)
# image_b = cv2.imread('data_gen/train/basset/8282ea64f812be023f8ad901385d66ba.jpg',0)
#
# images = [image_a,image_b]
# features = train_findFeatures(images)
# print("features",features)




# img = cv2.imread("train/0a1b0b7df2918d543347050ad8b16051.jpg",cv2.IMREAD_GRAYSCALE)
# cv2.imshow("image",img)
# print("image",img.shape)
# print(type(img))
# print("==========")
# img_1 = cv2.resize(img,(200,200))
# print(img_1.shape)
# cv2.imshow("image_1",img_1)
# cv2.waitKey(3000)
# cv2.destroyAllWindows()
# kp1, des1 = cv2.xfeatures2d.SURF_create().detectAndCompute(img_1,None);
#
#
# print("kp1:", type(kp1))
# print("==========")
# print(len(kp1))
# print(np.array(kp1).shape)
# print("==========")
# print(kp1)
#
# print("des1",type(des1))
# print("==========")
# print(des1.shape)
# print("==========")
# print(des1)
#
# k = 100;
# voc, variance = kmeans(des1, k, 1)
#
# print("voc:", type(voc))
# print("==========")
# print(np.array(voc).shape)
# print("==========")
# print(voc)
#
# print("variance:", type(variance))
# print("==========")
# print(np.array(variance).shape)
# print("==========")
# print(variance)
# ===========================

#
#
# from __future__ import print_function
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import numpy as np
# from Calculate_Accuracy import cal_Accuracy
#
# iris = datasets.load_iris()
# iris_X = iris.data
# iris_y = iris.target
#
#
# print("=========")
# X_train, X_test, y_train, y_test = train_test_split(
#     iris_X, iris_y, test_size=0.3)
#
# dic = {0:"andy_0",1:"tomming_1",2:"iris_2"}
# print("y_train",y_train)
#
# y_train1 = []
# for i in range(0,len(y_train)):
#     y_train1.append(dic[y_train[i]])
#
# print(y_train1)
# print("=======================================")
# rf = RandomForestClassifier(n_estimators=10,
#                  criterion="gini",
#                  max_depth=None,
#                  min_samples_split=2,
#                  min_samples_leaf=1,
#                  min_weight_fraction_leaf=0.,
#                  max_features="auto",
#                  max_leaf_nodes=None,
#                  min_impurity_decrease=0.,
#                  min_impurity_split=None,
#                  bootstrap=True,
#                  oob_score=False,
#                  n_jobs=1,
#                  random_state=None,
#                  verbose=0,
#                  warm_start=False,
#                  class_weight=None)
# rf.fit(X_train,y_train1)
#
# # knn.fit(X_train, y_train)
# predict = rf.predict(X_test)
#
# print("predict",predict)
# print("==============")
# print("y_test ",y_test)
# #
# print("accracy is:")
# print(cal_Accuracy(predict,y_test))



# ========================================

import pandas as pd
from creatSubmition import creatSubmition
from tqdm import tqdm
import numpy as np

df_train = pd.read_csv("./labels.csv")

df_predict = pd.read_csv("./test_submission.csv")
df_test = pd.read_csv('./sample_submission.csv')

creatSubmition(df_predict, df_train, df_test, "1122.csv")




