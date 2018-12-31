# import the necessary packages
from config import tiny_imagenet_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.preprocessing import SimplePreprocessor
from pyimagesearch.preprocessing import MeanPreprocessor
from pyimagesearch.utils.ranked import rank5_accuracy
from pyimagesearch.io import HDF5DatasetGenerator
from keras.models import load_model
import json

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(64, 64)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the testing dataset generator
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64,
                               preprocessors=[sp, mp, iap], 
                               classes=config.NUM_CLASSES)


# load the pre-trained network
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# make predictions on the testing data
print("[INFO] predicting on test data...")
predictions = model.predict_generator(testGen.generator(),
                                      steps=testGen.numImages // 64,
                                      max_queue_size=64 * 2)


# compute the rank-1 and rank-5 accuracies
(rank1, rank5) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
print("[INFO] rank-5: {:.2f}%".format(rank5 * 100))

# close the database
testGen.close()