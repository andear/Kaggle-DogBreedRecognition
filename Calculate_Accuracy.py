def cal_Accuracy(predict, test):
    if len(predict) != len(test):
        return -1;

    same = 0
    for i in range(len(predict)):
        if predict[i] == test[i]:
            same = same + 1

    return same/len(predict)