# import the necessary packages
from config import age_gender_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.utils import AgeGenderHelper
import numpy as np
import progressbar
import pickle
import json

import cv2
# initialize our helper class, then build the set of image paths
# and class labels
print("[INFO] building paths and labels...")
agh = AgeGenderHelper(config)
(trainPaths, trainLabels) = agh.buildPathsAndLabels()
# now that we have the total number of images in the dataset that
# can be used for training, compute the number of images that
# should be used for validation and testing
numVal = int(len(trainPaths) * config.NUM_VAL_IMAGES)
numTest = int(len(trainPaths) * config.NUM_TEST_IMAGES)
# our class labels are represented as strings so we need to encode
# them
print("[INFO] encoding labels...")
le = LabelEncoder().fit(trainLabels)
trainLabels = le.transform(trainLabels)

# perform sampling from the training set to construct a a validation
# set
print("[INFO] constructing validation data...")
split = train_test_split(trainPaths, trainLabels, test_size=numVal,
stratify=trainLabels)
(trainPaths, valPaths, trainLabels, valLabels) = split

# perform stratified sampling from the training set to construct a
# a testing set
print("[INFO] constructing testing data...")
split = train_test_split(trainPaths, trainLabels, test_size=numTest,
stratify=trainLabels)
(trainPaths, testPaths, trainLabels, testLabels) = split

# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output list
# files
datasets = [
("train", trainPaths, trainLabels, config.TRAIN_MX_LIST),
("val", valPaths, valLabels, config.VAL_MX_LIST),
("test", testPaths, testLabels, config.TEST_MX_LIST)]
# initialize the lists of RGB channel averages
(R, G, B) = ([], [], [])

# loop over the dataset tuples
for (dType, paths, labels, outputPath) in datasets:
# open the output file for writing
print("[INFO] building {}...".format(outputPath))
f = open(outputPath, "w")
# initialize the progress bar
widgets = ["Building List: ", progressbar.Percentage(), " ",
progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(paths),
widgets=widgets).start()

# loop over each of the individual images + labels
for (i, (path, label)) in enumerate(zip(paths, labels)):
# if we are building the training dataset, then compute the
# mean of each channel in the image, then update the
# respective lists
if dType == "train":
image = cv2.imread(path)
(b, g, r) = cv2.mean(image)[:3]
R.append(r)
G.append(g)
B.append(b)
# write the image index, label, and output path to file
row = "\t".join([str(i), str(label), path])
f.write("{}\n".format(row))
pbar.update(i)
# close the output file
pbar.finish()
f.close()

# construct a dictionary of averages, then serialize the means to a
# JSON file
print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()
# serialize the label encoder
print("[INFO] serializing label encoder...")
f = open(config.LABEL_ENCODER_PATH, "wb")
f.write(pickle.dumps(le))
f.close()