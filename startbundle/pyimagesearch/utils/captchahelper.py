import imutils
import cv2


def preprocess(image, width, height):
    # Grab the dimensions of the image, then initialize the padding values
    (h, w) = image.shape[:2]

    # If the width is greater than the height then resize along the width
    if w > h:
        image = imutils.resize(image, width=width)
    # Otherwise, the height is greater than the width so resize along the height
    else:
        image = imutils.resize(image, height=height)

    # Determine the padding values for the width and height to obtain the target dimensions
    pad_w = int((width - image.shape[1]) / 2.0)
    pad_h = int((height - image.shape[0]) / 2.0)

    # Pad the image then apply one more resizing to handle any rounding issues
    image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # Return the pre-processed image
    return image
