import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

def extract_training_features():
    data_dir = 'data_gen/'
    contents = os.listdir(data_dir)
    classes = [each for each in contents if os.path.isdir(data_dir + each)]

    batch_size = 10
    codes_list = []
    labels = []
    batch = []
    codes = None

    with tf.Session() as sess:
        # Construct VGG16 object
        vgg = vgg16.Vgg16()
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with tf.name_scope("content_vgg"):
            # build VGG16 model
            vgg.build(input_)

        # for every kind of flower, use VGG16 to calculate its feature
        for each in classes:
            print("Starting {} images".format(each))
            class_path = data_dir + each
            files = os.listdir(class_path)
            for ii, file in enumerate(files, 1):
                # 载入图片并放入batch数组中
                img = utils.load_image(os.path.join(class_path, file))
                batch.append(img.reshape((1, 224, 224, 3)))
                labels.append(each)

                # 如果图片数量到了batch_size则开始具体的运算
                if ii % batch_size == 0 or ii == len(files):
                    images = np.concatenate(batch)

                    feed_dict = {input_: images}
                    # 计算特征值
                    codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

                    # 将结果放入到codes数组中
                    if codes is None:
                        codes = codes_batch
                    else:
                        codes = np.concatenate((codes, codes_batch))

                    # 清空数组准备下一个batch的计算
                    batch = []
                    print('{} images processed'.format(ii))

    np.save("train_features.npy",codes)
    np.save("train_labels.npy",np.array(labels))

def extract_testing_features():
    batch_size = 10
    batch = []
    codes = None

    df_test = pd.read_csv('./sample_submission.csv')

    with tf.Session() as sess:
        # Construct VGG16 object
        vgg = vgg16.Vgg16()
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with tf.name_scope("content_vgg"):
            # build VGG16 model
            vgg.build(input_)

        files = df_test['id'].values
        for ii, file in enumerate(files, 1):
            # 载入图片并放入batch数组中
            img = utils.load_image('./test/{}.jpg'.format(file))
            batch.append(img.reshape((1, 224, 224, 3)))

            # 如果图片数量到了batch_size则开始具体的运算
            if ii % batch_size == 0 or ii == len(files):
                images = np.concatenate(batch)

                feed_dict = {input_: images}
                # 计算特征值
                codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

                # 将结果放入到codes数组中
                if codes is None:
                    codes = codes_batch
                else:
                    codes = np.concatenate((codes, codes_batch))

                # 清空数组准备下一个batch的计算
                batch = []
                print('{} images processed'.format(ii))

    np.save("test_features.npy", codes)

if __name__ == '__main__':
    # extract_training_features()
    extract_testing_features()