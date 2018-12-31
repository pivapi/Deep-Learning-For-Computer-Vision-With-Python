# import the necessary packages
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sys
sys.path.append("..")
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from imutils import paths
import argparse

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
args = vars(ap.parse_args())

# Get list of image paths
imagePaths = list(paths.list_images(args['dataset']))

# initilize the image preprocessor, load the dataset from disk.
# and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=5)

# loop over our set of regularizaers
for r in (None, "l1", "l2", "elasticnet"):
    print("[INFO] training model with '{}' penalty".format(r))
    model = SGDClassifier(loss="log", penalty=r, max_iter=10, learning_rate="constant", eta0=0.01, random_state=42)
    model.fit(trainX, trainY)

    # eveluate the classifier
    acc = model.score(testX, testY)
    print("[INFO] '{}' penalty accuracy:{:.2}%".format(r, acc * 100))

