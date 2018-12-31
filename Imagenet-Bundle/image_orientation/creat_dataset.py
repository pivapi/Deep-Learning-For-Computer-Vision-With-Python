# import the necessary packages
from imutils import paths
import numpy as np
import progressbar
import argparse
import imutils
import random
import cv2
import os
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input directory of images")
ap.add_argument("-o", "--output", required=True,
                help="path to output directory of rotated iamges")
args = vars(ap.parse_args())

# grab the paths to the input images (limiting ourselves to 10,000
# images) and shuffle them to make creating a training and testing
# split easier
imagePaths = list(paths.list_images(args["dataset"]))[:10000]
random.shuffle(imagePaths)
# initialize a dictionary to keep track of the number of each angle
# chosen so far, then initialize the progress bar
angles = {}
widgets = ["Building Dataset: ", progressbar.Percentage(), " ",
           progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),
                               widgets=widgets).start()

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # determine the rotation angle, and load the image
    angle = np.random.choice([0, 90, 180, 270])
    image = cv2.imread(imagePath)
    # if the image is None (meaning there was an issue loading the
    # image from disk, simply skip it)
    if image is None:
        continue

    # rotate the image based on the selected angle, then construct
    # the path to the base output directory
    image = imutils.rotate_bound(image, angle)
    base = os.path.sep.join([args["output"], str(angle)])
    # if the base path does not exist already, create it
    if not os.path.exists(base):
        os.makedirs(base)
    # extract the image file extension, then construct the full path
    # to the output file
    ext = imagePath[imagePath.rfind("."):]
    outputPath = [base, "image_{}{}".format(str(angles.get(angle, 0)).zfill(5), ext)]
    outputPath = os.path.sep.join(outputPath)
    # save the image
    cv2.imwrite(outputPath, image)

    # update the count for the angle
    c = angles.get(angle, 0)
    angles[angle] = c + 1
    pbar.update(i)

# finish the progress bar
pbar.finish()
# loop over the angles and display counts for each of them
for angle in sorted(angles.keys()):
    print("[INFO] angle={}: {:,}".format(angle, angles[angle]))