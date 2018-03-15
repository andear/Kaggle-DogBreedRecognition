import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from creatSubmition import creatSubmition

start = time.clock()

train_features = np.load("train_features.npy")
temp = np.load("train_labels.npy")
labels = np.ndarray.tolist(temp)

test_features = np.load("test_features.npy")

nbr_occurences = np.sum((train_features > 0) * 1, axis=0)

idf = np.array(np.log((1.0 * len(train_features) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

train_features = idf * train_features;
test_features = idf * test_features

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

RF_train_start = time.clock()

print("\nStart training RF...")
rf.fit(train_features, labels)
RF_train_end = time.clock()
print("\ntraining done ! Training Time:", RF_train_end - RF_train_start)
RF_test_start = time.clock()
print("\nStart predict...")
predict = rf.predict(test_features)
print("\npredict done! predict Time:", time.clock() - RF_test_start)
#
df_predict = pd.DataFrame(predict)
df_predict.to_csv("predict.csv")
df_predict.columns = ['predict']

df_train = pd.read_csv("./labels.csv")
df_test = pd.read_csv('./sample_submission.csv')

end = time.clock()
print('Total time:', end - start)


fileName = "transfer_learning.csv"
creatSubmition(df_predict, df_train, df_test, fileName=fileName)
# acc = cal_Accuracy(predict,test_classes)
# print("accuracy is", acc)