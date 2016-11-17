# import the necessary packages
import pickle
import cv2
from rgbhistogram import RGBHistogram
from moviebox import find_box

def index_movie(inputFile, outputFile):
	# construct the argument parser and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--input", required = True,
	# 	help = "Path to the movie to be indexed")
	# ap.add_argument("-o", "--output", required = True,
	# 	help = "Path to where the computed index will be stored")
	# args = vars(ap.parse_args())

	# initialize the index dictionary to store our our quantifed
	# images, with the 'key' of the dictionary being the image
	# filename and the 'value' our computed features
	index = {}

	# initialize our image descriptor -- a 3D RGB histogram with
	# 8 bins per channel
	desc = RGBHistogram([8, 8, 8])

	cap = cv2.VideoCapture(inputFile)
	index[-1] = cap.get(cv2.CAP_PROP_FPS)
	count = 0

	x, y, w, h = find_box(inputFile)

	while cap.isOpened():
		ret, frame = cap.read()
		if ret:
		    # load the image, describe it using our RGB histogram
			# descriptor, and update the index
			features = desc.describe(frame[y:y+h, x:x+w])
			index[count] = features
			count = count + 1
		else:
			break

# we are now done indexing our image -- now we can write our
# index to disk
	f = open(outputFile, "wb")

	f.write(pickle.dumps(index))
	f.close()
	return index
