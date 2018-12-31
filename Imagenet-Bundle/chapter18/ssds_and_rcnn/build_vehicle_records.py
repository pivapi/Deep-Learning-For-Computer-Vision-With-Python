# import the necessary packages
from config import dlib_front_rear_config as config
from pyimagesearch.utils.tfannotation import TFAnnotation
from bs4 import BeautifulSoup
from PIL import Image
import tensorflow as tf
import os

def main(_):
    # open the classes output file
    f = open(config.CLASSES_FILE, "w")
    # loop over the classes
    for (k, v) in config.CLASSES.items():
        # construct the class information and write to file
        item = ("item {\n"
                "\tid: " + str(v) + "\n"
                "\tname: ’" + k + "’\n"
                "}\n")
        f.write(item)
    # close the output classes file
    f.close()

    # initialize the data split files
    datasets = [
        ("train", config.TRAIN_XML, config.TRAIN_RECORD),
        ("test", config.TEST_XML, config.TEST_RECORD)
    ]

    # loop over the datasets
    for (dType, inputPath, outputPath) in datasets:
        # build the soup
        print("[INFO] processing ’{}’...".format(dType))
        contents = open(inputPath).read()
        soup = BeautifulSoup(contents, "html.parser")
        # initialize the TensorFlow writer and initialize the total
        # number of examples written to file
        writer = tf.python_io.TFRecordWriter(outputPath)
        total = 0

        # loop over all image elements
        for image in soup.find_all("image"):
            # load the input image from disk as a TensorFlow object
            p = os.path.sep.join([config.BASE_PATH, image["file"]])
            encoded = tf.gfile.GFile(p, "rb").read()
            encoded = bytes(encoded)
            # load the image from disk again, this time as a PIL
            # object
            pilImage = Image.open(p)
            (w, h) = pilImage.size[:2]
            # parse the filename and encoding from the input path
            filename = image["file"].split(os.path.sep)[-1]
            encoding = filename[filename.rfind(".") + 1:]
            # initialize the annotation object used to store
            # information regarding the bounding box + labels
            tfAnnot = TFAnnotation()
            tfAnnot.image = encoded
            tfAnnot.encoding = encoding
            tfAnnot.filename = filename
            tfAnnot.width = w
            tfAnnot.height = h

            # loop over all bounding boxes associated with the image
            for box in image.find_all("box"):
                # check to see if the bounding box should be ignored
                if box.has_attr("ignore"):
                    continue

                # extract the bounding box information + label,
                # ensuring that all bounding box dimensions fit
                # inside the image
                startX = max(0, float(box["left"]))
                startY = max(0, float(box["top"]))
                endX = min(w, float(box["width"]) + startX)
                endY = min(h, float(box["height"]) + startY)
                label = box.find("label").text

                # TensorFlow assumes all bounding boxes are in the
                # range [0, 1] so we need to scale them
                xMin = startX / w
                xMax = endX / w
                yMin = startY / h
                yMax = endY / h

                # due to errors in annotation, it may be possible
                # that the minimum values are larger than the maximum
                # values -- in this case, treat it as an error during
                # annotation and ignore the bounding box
                if xMin > xMax or yMin > yMax:
                    continue
                # similarly, we could run into the opposite case
                # where the max values are smaller than the minimum
                # values
                elif xMax < xMin or yMax < yMin:
                    continue

                # update the bounding boxes + labels lists
                tfAnnot.xMins.append(xMin)
                tfAnnot.xMaxs.append(xMax)
                tfAnnot.yMins.append(yMin)
                tfAnnot.yMaxs.append(yMax)
                tfAnnot.textLabels.append(label.encode("utf8"))
                tfAnnot.classes.append(config.CLASSES[label])
                tfAnnot.difficult.append(0)
                # increment the total number of examples
                total += 1

            # encode the data point attributes using the TensorFlow
            # helper functions
            features = tf.train.Features(feature=tfAnnot.build())
            example = tf.train.Example(features=features)
            # add the example to the writer
            writer.write(example.SerializeToString())

        # close the writer and print diagnostic information to the
        # user
        writer.close()
        print("[INFO] {} examples saved for ’{}’".format(total, dType))

# check to see if the main thread should be started
if __name__ == "__main__":
    tf.app.run()