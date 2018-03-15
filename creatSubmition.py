import pandas as pd
def creatSubmition(df_predict, df_train, df_test, fileName):
    temp_series = pd.Series(df_predict['predict'])

    # Set column names to those generated by the one-hot encoding earlier
    one_hot_predict = pd.get_dummies(temp_series, sparse=True)
    print(type(one_hot_predict))
    print(one_hot_predict.shape)

    # Insert the column id from the sample_submission at the start of the data frame
    one_hot_predict.insert(0, 'id', df_test['id'])
    one_hot_predict.to_dense().to_csv(fileName, index=False)

