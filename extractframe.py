# import the necessary packages
from moviebox import find_box
from rgbhistogram import RGBHistogram
import argparse
import pickle
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True,
	help = "Path to the movie to be extracted")
ap.add_argument("-f", "--frame", required = True,
	help = "Frame number to be extracted")
args = vars(ap.parse_args())
 
# initialize the index dictionary to store our our quantifed
# images, with the 'key' of the dictionary being the image
# filename and the 'value' our computed features
frameno=int(args["frame"])

# initialize our image descriptor -- a 3D RGB histogram with
# 8 bins per channel
desc = RGBHistogram([8, 8, 8])

cap = cv2.VideoCapture(args["input"])
count = 0

x,y,w,h = find_box(args["input"])


cap.set(1, frameno);
ret, frame = cap.read()
cv2.imwrite('frame'+str(frameno)+'.jpg', frame[y:y+h,x:x+w])