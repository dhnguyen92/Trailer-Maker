import cv2

NUM_REGION = 3;

class RGBHistogram:
	def __init__(self, bins):
		# store the number of bins the histogram will use
		self.bins = bins

	def describe(self, image):
		# compute a 3D histogram in the RGB colorspace,
		# then normalize the histogram so that images
		# with the same content, but either scaled larger
		# or smaller will have (roughly) the same histogram
		width = image.shape[1]
		height = image.shape[0]
		result = []

		for i in range(0, NUM_REGION):
			for j in range(0, NUM_REGION):
				hist = cv2.calcHist([image[i*height//NUM_REGION:(i+1)*height//NUM_REGION, j*width//NUM_REGION:(j+1)*width//NUM_REGION]], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])
				cv2.normalize(hist, hist)
				result.append(hist.flatten())

		# return out 3D histogram as a flattened array
		return hist.flatten()
