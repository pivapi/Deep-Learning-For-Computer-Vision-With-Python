# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import imutils
import cv2

image = cv2.imread('myself.jpg');
cv2.imshow('image',image)
cv2.waitKey(0)


def preprocess(image, width = 28, height = 28, inter=cv2.INTER_AREA):
    # grab the dimensions of the image and then initialize
    # the deltas to use when cropping
    (h, w) = image.shape[:2]
    dW = 0
    dH = 0

    # if the width is smaller than the height, then resize
    # along the width (i.e., the smaller dimension) and then
    # update the deltas to crop the height to the desired
    # dimension
    if w < h:
        image = imutils.resize(image, width= width, inter=inter)
        dH = int((image.shape[0] - height) / 2.0)

        # otherwise, the height is smaller than the width so
        # resize along the height and then update the deltas
        # to crop along the width
    else:
        image = imutils.resize(image, height=height, inter=inter)
        dW = int((image.shape[1] - width) / 2.0)

    cv2.imshow('image', image)
    cv2.waitKey(0)

        # re-grab the width and height, followed by performing
        # the crop
    (h, w) = image.shape[:2]
    image = image[dH:h - dH, dW:w - dW]

    cv2.imshow('image', image)
    cv2.waitKey(0)

    # finally, resize the image to the provided spatial
    # dimensions to ensure our output image is always a fixed
    # size
    return cv2.resize(image, (width, height), interpolation=inter)

image = preprocess(image)

cv2.imshow('image',image)
cv2.waitKey(0)