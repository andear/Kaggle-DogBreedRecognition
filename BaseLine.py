from Baseline_findFeatures import train_findFeatures, test_findFeatures
from Baseline_LoadImage import read_training_images, read_testing_images
from creatSubmition import creatSubmition
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn import svm

start = time.clock()
width = 200
height = 200
numberOfWords = 400
(images,classes,df_train) = read_training_images(img_width = width, img_height = height)


(training_features,voc,idf) = train_findFeatures(images,numWords = numberOfWords)

print("features:",type(training_features))
print("==========")
print("features.shape:",training_features.shape)
print("==========")


rf = RandomForestClassifier(n_estimators=1000,
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
# rf = svm.SVC()

RF_train_start = time.clock()
print("Time for extracting feature:", RF_train_start - start)

print("\nStart training RF...")
rf.fit(training_features,classes)
RF_train_end = time.clock()
print("\ntraining done ! Training Time:", RF_train_end - RF_train_start)



(test_images,df_test) = read_testing_images(img_width = width, img_height = height)


testing_features = test_findFeatures(test_images,voc,idf,numWords = numberOfWords)

print("\nStart predict...")
RF_test_start = time.clock()
predict = rf.predict(testing_features)
print("\npredict done! predict Time:", time.clock() - RF_test_start)

df_predict = pd.DataFrame(predict)
df_predict.columns = ['predict']
df_predict.to_csv("predict.csv")

# fileName = "baseline_k=" + numberOfWords + "_" + width + "_" +height + ".csv"

end = time.clock()
print('Total time:', end - start)


fileName = "baseline.csv"
creatSubmition(df_predict, df_train, df_test, fileName=fileName)



