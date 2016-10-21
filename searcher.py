# import the necessary packages
import numpy as np
 
FRAME_SKIP = 0.5

class Searcher:
	def __init__(self, index):
		# store our index of images
		self.index = index
 
	def search(self, queryFeatures, skipFrame=FRAME_SKIP, startFrame=0, endFrame=-1):
		# initialize our dictionary of results
		results = {}
		start_skip_frame=int(startFrame/(FRAME_SKIP*self.getFPS()))
		if endFrame==-1:
			endFrame = len(self.index)-1
		else :
			endFrame=min(endFrame, len(self.index)-1)

		end_skip_frame= int(endFrame/(FRAME_SKIP*self.getFPS()))
		# loop over the index
		for i in range(start_skip_frame,end_skip_frame):
			# compute the chi-squared distance between the features
			# in our index and our query features -- using the
			# chi-squared distance which is normally used in the
			# computer vision field to compare histograms
			curFrame = i*FRAME_SKIP*self.getFPS()
			d = self.chi2_distance(self.index[curFrame], queryFeatures)
 
			# now that we have the distance between the two feature
			# vectors, we can udpate the results dictionary -- the
			# key is the current image ID in the index and the
			# value is the distance we just computed, representing
			# how 'similar' the image in the index is to our query
			results[curFrame] = d
 
		# sort our results, so that the smaller distances (i.e. the
		# more relevant images are at the front of the list)
		results = sorted([(v, k) for (k, v) in results.items()])
 
		# return our results
		return results
 
	def chi2_distance(self, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])
 
		# return the chi-squared distance
		return d

	def getFPS(self):
		return round(self.index[-1])