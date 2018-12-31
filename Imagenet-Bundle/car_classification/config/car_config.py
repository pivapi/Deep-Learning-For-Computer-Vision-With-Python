# import the necessary packages
from os import path
# define the base path to the cars dataset
BASE_PATH = "/raid/datasets/cars"
# based on the base path, derive the images path and meta file path
IMAGES_PATH = path.sep.join([BASE_PATH, "car_ims"])
LABELS_PATH = path.sep.join([BASE_PATH, "complete_dataset.csv"])

# define the path to the output training, validation, and testing
# lists
MX_OUTPUT = BASE_PATH
TRAIN_MX_LIST = path.sep.join([MX_OUTPUT, "lists/train.lst"])
VAL_MX_LIST = path.sep.join([MX_OUTPUT, "lists/val.lst"])
TEST_MX_LIST = path.sep.join([MX_OUTPUT, "lists/test.lst"])

# define the path to the output training, validation, and testing
# image records
TRAIN_MX_REC = path.sep.join([MX_OUTPUT, "rec/train.rec"])
VAL_MX_REC = path.sep.join([MX_OUTPUT, "rec/val.rec"])
TEST_MX_REC = path.sep.join([MX_OUTPUT, "rec/test.rec"])

# define the path to the label encoder
LABEL_ENCODER_PATH = path.sep.join([BASE_PATH, "output/le.cpickle"])

# define the RGB means from the ImageNet dataset
R_MEAN = 123.68
G_MEAN = 116.779
B_MEAN = 103.939

# define the percentage of validation and testing images relative
# to the number of training images
NUM_CLASSES = 164
NUM_VAL_IMAGES = 0.15
NUM_TEST_IMAGES = 0.15

# define the batch size
BATCH_SIZE = 32
NUM_DEVICES = 1