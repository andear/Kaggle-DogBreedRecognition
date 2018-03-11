import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
# from imagePreprocess import pca_rebuild_img



def plot_images(images, classes,img_width, img_height):
    assert len(images) == len(classes) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3, figsize=(60, 60), sharex=True)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.

        ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB).reshape(img_width, img_height, 3), cmap='hsv')
        xlabel = "Breed: {0}".format(classes[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        ax.xaxis.label.set_size(60)
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.

    plt.show()

def read_training_images(img_width = 250, img_height = 250):

    df_train = pd.read_csv("./labels.csv")
    # print(df_train.head(10))

    images =[]
    classes =[]

    print("Loading Training Image...")
    for f, breed in tqdm(df_train.values):
        img = cv2.imread('./train/{}.jpg'.format(f), cv2.IMREAD_GRAYSCALE)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        # img = pca_rebuild_img(img,0.8)
        # img = cv2.imread('./train/{}.jpg'.format(f))
        classes.append(breed)
        images.append(cv2.resize(img, (img_width, img_height)))
    return (images,classes,df_train)

def read_testing_images(img_width = 250, img_height = 250):

    df_test = pd.read_csv('./sample_submission.csv')

    images = []
    print("Loading Testing Image...")
    for f in tqdm(df_test['id'].values):
        img = cv2.imread('./test/{}.jpg'.format(f), cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread('./test/{}.jpg'.format(f))
        img = cv2.GaussianBlur(img, (3, 3), 0)
        # img = pca_rebuild_img(img, 0.8)
        images.append(cv2.resize(img, (img_width, img_height)))
    return (images,df_test)
















# img = cv2.imread("train/0a1b0b7df2918d543347050ad8b16051.jpg",cv2.IMREAD_GRAYSCALE)
# cv2.imshow("image",img)
# print(img.shape)
# print(type(img))
# img_1 = cv2.resize(img,(200,200))
# print(img_1.shape)
# cv2.imshow("image_1",img_1)
# cv2.waitKey(3000)
# cv2.destroyAllWindows()