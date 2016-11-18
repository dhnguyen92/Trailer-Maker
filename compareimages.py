# import the necessary packages
import argparse
import cv2
import numpy as np
from rgbhistogram2 import RGBHistogram2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True,	help="Path to the first image")
ap.add_argument("-d", "--destination", required=True, help="Path to the second image")
args = vars(ap.parse_args())

src_image = cv2.imread(args["source"])
dest_image = cv2.imread(args["destination"])

desc = RGBHistogram2([8, 8, 8])
src_features = desc.describe(src_image)
dest_features = desc.describe(dest_image)

# print(src_features)

eps = 1e-10
result = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(src_features, dest_features)])

print(result)
