# import cv2
# from matplotlib import pyplot as plt
# import numpy as np
# from scipy.cluster.vq import *
#
# img = cv2.imread("train/0a1b0b7df2918d543347050ad8b16051.jpg",cv2.IMREAD_GRAYSCALE)
# cv2.imshow("image",img)
# print("image",img.shape)
# print(type(img))
# print("==========")
# img_1 = cv2.resize(img,(200,200))
# # print(img_1.shape)
# # cv2.imshow("image_1",img_1)
# # cv2.waitKey(3000)
# # cv2.destroyAllWindows()
#
# kp1, des1 = cv2.xfeatures2d.SURF_create().detectAndCompute(img_1,None);
#
#
# # print("kp1:", type(kp1))
# # print("==========")
# # print(np.array(kp1).shape)
# # print("==========")
# # print(kp1)
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

# import pandas as pd
# from tqdm import tqdm
# import numpy as np
#
# df_train = pd.read_csv("./labels.csv")
# print(df_train.head(10))
#
# classes = []
#
# print("Loading Training Image...")
# for f, breed in tqdm(df_train.values):
#     classes.append(breed)
#
#
# print("==========")
# print("classes", type(classes))
# print("==========")
# print(np.array(classes).shape)
# print("==========")
# print(classes)


# ========================================
import numpy as np

a = np.array([1,2,3])
b = np.array([2,2,3])
c = np.array([3,2,3])
d = np.array([4,2,3])
temp1 = np.vstack((a,b))
temp2 = np.vstack((c,d))
result1 = np.vstack((temp1,temp2))

print(result1)

print("=========")
temp3 = np.vstack((temp1,c))
result2 = np.vstack((temp3,d))
print(result2)