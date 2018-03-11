from Baseline_findFeatures import train_findFeatures, test_findFeatures
from Baseline_LoadImage import read_training_images, read_testing_images
import numpy as np
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


def cal_Accuracy(predict, test):
    if len(predict) != len(test):
        return -1;

    same = 0
    for i in range(len(predict)):
        if predict[i] == test[i]:
            same = same + 1

    return same/len(predict)

def baseline():
    start = time.clock()
    width = 200
    height = 200
    numberOfWords = 100
    (images, classes, df_train) = read_training_images(img_width=width, img_height=height)

    # split to train and test
    train_images, test_images, train_classes, test_classes = train_test_split(images, classes, test_size=0.05)

    # X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
    # y is classes


    (train_features, voc, idf) = train_findFeatures(train_images, numWords=numberOfWords)

    test_features = test_findFeatures(test_images, voc,idf, numWords=numberOfWords)



    print("==========")
    print("train_features.shape:",train_features.shape)
    print("==========")
    print("test_features.shape:", test_features.shape)
    print("==========")

    rf = RandomForestClassifier(n_estimators=500,
                     criterion="gini",
                     max_depth=None,
                     min_samples_split=2,
                     min_samples_leaf=1,
                     min_weight_fraction_leaf=0.,
                     # max_features="sqrt",
                     max_features="auto",
                     max_leaf_nodes=None,
                     min_impurity_decrease=0.,
                     min_impurity_split=None,
                     bootstrap=True,
                     oob_score=False,
                     n_jobs=1,
                     random_state=None,
                     verbose=0,
                     warm_start=False,
                     class_weight=None)
    # rf1 = svm.SVC()
    # rf = AdaBoostClassifier(n_estimators=100)
    # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth = 1, random_state = 0)
    # print("Cross validation...")
    # scores = cross_val_score(rf, train_features, train_classes)
    # print("scores:", scores)

    RF_train_start = time.clock()
    print("Time for extracting feature:", RF_train_start - start)

    print("\nStart training RF...")
    rf.fit(train_features,train_classes)
    # rf1.fit(train_features, train_classes)
    RF_train_end = time.clock()
    print("\ntraining done ! Training Time:", RF_train_end - RF_train_start)

    # print("scores for SVM:", rf1.score(test_features, test_classes))

    print("\nStart predict...")
    RF_test_start = time.clock()
    predict = rf.predict(test_features)
    print("\npredict done! predict Time:", time.clock() - RF_test_start)

    print("==========")
    print("train_features.shape:", predict.shape)
    print("==========")

    # print("scores:",rf.score(test_features,test_classes))

    acc = cal_Accuracy(predict,test_classes)
    print("accuracy is", acc)


def transfer_learning():
    start = time.clock()

    features = np.load("train_features.npy")
    temp = np.load("train_labels.npy")
    labels = np.ndarray.tolist(temp)

    train_features, test_features, train_classes, test_classes = train_test_split(features, labels, test_size=0.05)


    rf = RandomForestClassifier(n_estimators=100,
                     criterion="gini",
                     max_depth=None,
                     min_samples_split=2,
                     min_samples_leaf=1,
                     min_weight_fraction_leaf=0.,
                     # max_features="sqrt",
                     max_features="auto",
                     max_leaf_nodes=None,
                     min_impurity_decrease=0.,
                     min_impurity_split=None,
                     bootstrap=True,
                     oob_score=False,
                     n_jobs=1,
                     random_state=None,
                     verbose=0,
                     warm_start=False,
                     class_weight=None)

    RF_train_start = time.clock()
    print("Time for extracting feature:", RF_train_start - start)

    print("\nStart training RF...")
    rf.fit(train_features,train_classes)
    RF_train_end = time.clock()
    print("\ntraining done ! Training Time:", RF_train_end - RF_train_start)
    RF_test_start = time.clock()
    print("\nStart predict...")
    print("scores for Random Forest:", rf.score(test_features, test_classes))
    # predict = rf.predict(test_features)
    print("\npredict done! predict Time:", time.clock() - RF_test_start)
    #
    # acc = cal_Accuracy(predict,test_classes)
    # print("accuracy is", acc)
if __name__ == '__main__':
    # baseline()
    transfer_learning()