# import the necessary packages
import os

# initialize the base path for the front/rear vehicle dataset
BASE_PATH = "dlib_front_and_rear_vehicles_v1"

# build the path to the input training and testing XML files
TRAIN_XML = os.path.sep.join([BASE_PATH, "training.xml"])
TEST_XML = os.path.sep.join([BASE_PATH, "testing.xml"])

# build the path to the output training and testing record files,
# along with the class labels file
TRAIN_RECORD = os.path.sep.join([BASE_PATH,
                                 "records/training.record"])
TEST_RECORD = os.path.sep.join([BASE_PATH,
                                "records/testing.record"])
CLASSES_FILE = os.path.sep.join([BASE_PATH,
                                 "records/classes.pbtxt"])

# initialize the class labels dictionary
CLASSES = {"rear": 1, "front": 2}