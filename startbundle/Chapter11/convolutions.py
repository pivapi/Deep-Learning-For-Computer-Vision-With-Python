
# import the necessary packages
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2


def convolve(image, K):
    # grab the spatial dimensions of the image and kernel
    (iH, iW) = image.shape[:2]
    (kK, kW) = K.shape[:2]

    # allocate memory for the output image, taking care to "pad"
    # the broders of the input image so the spatial size are not reduced
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    output = np.zeros((iH, iW), dtype="float")

    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y) -coordinates
            # dimensions
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

            # perform the actual convolution by taking the
            # element-wise multiplication between the ROI and
            # the kernel, then summing the matrix
            k = (roi * K).sum()

            # store the convolved value in the output (x, y)-coordinated
            # of the output image
            output[y - pad, x - pad] = k

            # rescale the output image to be in the range [0, 256]
            output = rescale_intensity(output, in_range=(0, 255))
            output = (output * 255).astype("uint8")

            # return the output image
            return output

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 * (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]), dtype="int")

# construct the Laplacian kernel used to detect edge-like
# regions of an image
laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0],), dtype="int"
)

# construct the Sobel x-axis kernel
sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="int"
)

# construct the Sobel y-axis kernel
sobelY = np.array((
    [-1, -2, -1],
    [0, 0, 0],
    [-1, 2, 1]), dtype="int"
)

# construct an emboss kernel
emboss = np.array((
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]), dtype="int"
)

kernalBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", sobelY),
    ("emboss", emboss)
)

image = cv2.imread(args["image"])
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Loop over the kernels
for (kernel_name, kernel) in kernalBank:
    # Apply the kernel to the greyscale image using both 'convolve' functions
    print('[INFO]: Applying {} kernel'.format(kernel_name))
    convolve_output = convolve(grey, kernel)
    opencv_output = cv2.filter2D(grey, -1, kernel)

    # Show the output image
    cv2.imshow('Original', grey)
    cv2.imshow('{} - convolve'.format(kernel_name), convolve_output)
    cv2.imshow('{} - filter2D'.format(kernel_name), opencv_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

