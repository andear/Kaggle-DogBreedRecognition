import cv2
from matplotlib import pyplot as plt


def bgr_rgb(img):
    (r, g, b) = cv2.split(img)
    return cv2.merge([b, g, r])
def orb_detect(image_a, image_b):
    # feature match
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(image_a, None)
    kp2, des2 = orb.detectAndCompute(image_b, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(image_a, kp1, image_b, kp2, matches[:100], None, flags=2)

    return bgr_rgb(img3)

def sift_detect(img1, img2, detector='surf'):
    if detector.startswith('si'):
        print
        "sift detector......"
        sift = cv2.xfeatures2d.SURF_create()
    else:
        print
        "surf detector......"
        sift = cv2.xfeatures2d.SURF_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = [[m] for m, n in matches if m.distance < 0.7 * n.distance]

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    return bgr_rgb(img3)

image_a = cv2.imread("data_gen/train/basset/4ff1a1c62acccfbb0bb1a6ecb4f29682.jpg",0)
image_b = cv2.imread('data_gen/train/basset/8282ea64f812be023f8ad901385d66ba.jpg',0)

# image_c = cv2.imread("train/0120343862761d052d6a7ade81625c94.jpg",0)
image_c = cv2.imread('train/02b1c50fb7315423a664f3ce68c94e30.jpg')
image_c1 = cv2.resize(image_c,(100,100))
image_d = cv2.imread('train/02b1c50fb7315423a664f3ce68c94e30.jpg',0)


img2 = sift_detect(image_a, image_b)

plt.imshow(img2)
plt.show()