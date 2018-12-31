# import the necessary packages
from object_detection.utils.dataset_util import bytes_list_feature
from object_detection.utils.dataset_util import float_list_feature
from object_detection.utils.dataset_util import int64_list_feature
from object_detection.utils.dataset_util import int64_feature
from object_detection.utils.dataset_util import bytes_feature

class TFAnnotation:
    def __init__(self):
        # initialize the bounding box + label lists
        self.xMins = []
        self.xMaxs = []
        self.yMins = []
        self.yMaxs = []
        self.textLabels = []
        self.classes = []
        self.difficult = []
        
        # initialize additional variables, including the image
        # itself, spatial dimensions, encoding, and filename
        self.image = None
        self.width = None
        self.height = None
        self.encoding = None
        self.filename = None

    def build(self):
        # encode the attributes using their respective TensorFlow
        # encoding function
        w = int64_feature(self.width)
        h = int64_feature(self.height)
        filename = bytes_feature(self.filename.encode("utf8"))
        encoding = bytes_feature(self.encoding.encode("utf8"))
        image = bytes_feature(self.image)
        xMins = float_list_feature(self.xMins)
        xMaxs = float_list_feature(self.xMaxs)
        yMins = float_list_feature(self.yMins)
        yMaxs = float_list_feature(self.yMaxs)
        textLabels = bytes_list_feature(self.textLabels)
        classes = int64_list_feature(self.classes)
        difficult = int64_list_feature(self.difficult)

        # construct the TensorFlow-compatible data dictionary
        data = {
            "image/height": h,
            "image/width": w,
            "image/filename": filename,
            "image/source_id": filename,
            "image/encoded": image,
            "image/format": encoding,
            "image/object/bbox/xmin": xMins,
            "image/object/bbox/xmax": xMaxs,
            "image/object/bbox/ymin": yMins,
            "image/object/bbox/ymax": yMaxs,
            "image/object/class/text": textLabels,
            "image/object/class/label": classes,
            "image/object/difficult": difficult,
        }
        
        # return the data dictionary
        return data