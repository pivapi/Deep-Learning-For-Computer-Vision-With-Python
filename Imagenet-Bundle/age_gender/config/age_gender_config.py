# import the necessary packages
from os import path
# define the type of dataset we are training (i.e., either "age" or
# "gender")
DATASET_TYPE = "gender"
# define the base paths to the faces dataset and output path
BASE_PATH = "/raid/datasets/adience"
OUTPUT_BASE = "output"
MX_OUTPUT = BASE_PATH
# based on the base path, derive the images path and folds path
IMAGES_PATH = path.sep.join([BASE_PATH, "aligned"])
LABELS_PATH = path.sep.join([BASE_PATH, "folds"])

# define the percentage of validation and testing images relative
# to the number of training images
NUM_VAL_IMAGES = 0.15
NUM_TEST_IMAGES = 0.15

# define the batch size
BATCH_SIZE = 128
NUM_DEVICES = 2

# check to see if we are working with the "age" portion of the
# dataset
if DATASET_TYPE == "age":
    # define the number of labels for the "age" dataset, along with
    # the path to the label encoder
    NUM_CLASSES = 8
    LABEL_ENCODER_PATH = path.sep.join([OUTPUT_BASE,
                                        "age_le.cpickle"])
    # define the path to the output training, validation, and testing
    # lists
    TRAIN_MX_LIST = path.sep.join([MX_OUTPUT, "lists/age_train.lst"])
    VAL_MX_LIST = path.sep.join([MX_OUTPUT, "lists/age_val.lst"])
    TEST_MX_LIST = path.sep.join([MX_OUTPUT, "lists/age_test.lst"])
    # define the path to the output training, validation, and testing
    # image records
    TRAIN_MX_REC = path.sep.join([MX_OUTPUT, "rec/age_train.rec"])
    VAL_MX_REC = path.sep.join([MX_OUTPUT, "rec/age_val.rec"])
    TEST_MX_REC = path.sep.join([MX_OUTPUT, "rec/age_test.rec"])
    # derive the path to the mean pixel file
    DATASET_MEAN = path.sep.join([OUTPUT_BASE,
                                  "age_adience_mean.json"])

# otherwise, check to see if we are performing "gender"
# classification
elif DATASET_TYPE == "gender":
    # define the number of labels for the "gender" dataset, along
    # with the path to the label encoder
    NUM_CLASSES = 2
    LABEL_ENCODER_PATH = path.sep.join([OUTPUT_BASE,
                                        "gender_le.cpickle"])
    # define the path to the output training, validation, and testing
    # lists
    TRAIN_MX_LIST = path.sep.join([MX_OUTPUT,
                                   "lists/gender_train.lst"])
    VAL_MX_LIST = path.sep.join([MX_OUTPUT,
                                 "lists/gender_val.lst"])
    TEST_MX_LIST = path.sep.join([MX_OUTPUT,
                                  "lists/gender_test.lst"])
    # define the path to the output training, validation, and testing
    # image records
    TRAIN_MX_REC = path.sep.join([MX_OUTPUT, "rec/gender_train.rec"])
    VAL_MX_REC = path.sep.join([MX_OUTPUT, "rec/gender_val.rec"])
    TEST_MX_REC = path.sep.join([MX_OUTPUT, "rec/gender_test.rec"])
    # derive the path to the mean pixel file
    DATASET_MEAN = path.sep.join([OUTPUT_BASE,
                                  "gender_adience_mean.json"])