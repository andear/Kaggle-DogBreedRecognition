from findFeatures import train_findFeatures, test_findFeatures
from LoadImage import read_training_images, read_testing_images
from creatSubmition import creatSubmition
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time

start = time.clock()
(images,classes,df_train) = read_training_images(img_width = 200, img_height = 200)

(training_features,voc) = train_findFeatures(images,numWords = 2000)

print("features:",type(training_features))
print("==========")
print("features.shape:",training_features.shape)
print("==========")


rf = RandomForestClassifier(n_estimators=500,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="sqrt",
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
rf.fit(training_features,classes)
RF_train_end = time.clock()
print("\ntraining done ! Training Time:", RF_train_end - RF_train_start)



(test_images,df_test) = read_testing_images(img_width = 200, img_height = 200)

testing_features = test_findFeatures(test_images,voc,numWords = 2000)
print("\nStart predict...")
predict = rf.predict(testing_features)
print("\npredict done!")

df_predict = pd.DataFrame(predict)
creatSubmition(df_predict,df_train,df_test,"baseline.csv")

end = time.clock()
print('Total time:', end - start)

