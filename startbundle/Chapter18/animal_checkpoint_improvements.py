# python shallownet_animals.py  --dataset ../datasets/animals/

import sys
sys.path.append("..")
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.datasets import SimpleDatasetLoader
from pyimagesearch.nn.conv import ShallowNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pyimagesearch.nn.conv import MiniVGGNet
from keras.callbacks import ModelCheckpoint
import os
from pyimagesearch.nn.conv import LeNet


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-w", "--weights", required=True, help="path to weights directory")
args = vars(ap.parse_args())

# grab the list of images that we'll describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))


# initialize the image preprocessor
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0


# pritition the into training and testing splits using 75% of
# the data for training and the remaining 25% fir testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# construct the callback to save only the *best* model to disk
# based on the validation loss
fname = os.path.sep.join([args["weights"], "weights-{epoch:03d}-{val_acc:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_acc", mode="max", save_best_only=True, verbose=2)
callbacks = [checkpoint]

# train the network
print("[INFO] training network....")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=64,
              epochs=40,
              callbacks=callbacks,
              verbose=2)












