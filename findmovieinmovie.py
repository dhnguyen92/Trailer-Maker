# import the necessary packages
from searcher import Searcher
from rgbhistogram import RGBHistogram
from pathlib import Path
from indexmovie import index_movie
from moviebox import find_box
import numpy as np
import argparse
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--trailer", required = True,
	help = "Path to the trailer video")
ap.add_argument("-m", "--movie", required = True,
	help = "Path to the movie")
ap.add_argument("-s", "--skip", required = True,
	help = "Duration to skip checking")
args = vars(ap.parse_args())

movie_index_filename=args["movie"].rsplit( ".", 1 )[ 0 ]+'.db'
index_file = Path(movie_index_filename)
if index_file.is_file():
	print("Loading movie index from "+movie_index_filename)
	movie_index = pickle.load(open(movie_index_filename, 'rb'))
else:
	print("Index file not found, indexing movie...")
	movie_index = indexMovie(args["movie"], movie_index_filename)

print("Done")

movie_searcher = Searcher(movie_index)
	# results = searcher.search(queryFeatures)

interval = 1

trailer_cap = cv2.VideoCapture(args["trailer"])
movie_cap = cv2.VideoCapture(args["movie"])
fps = round(trailer_cap.get(cv2.CAP_PROP_FPS))
length = int(trailer_cap.get(cv2.CAP_PROP_FRAME_COUNT))
x,y,w,h = find_box(args["trailer"])

desc = RGBHistogram([8, 8, 8])
mapping=[]
threshold=0.1
skip_duration = float(args["skip"])

for i in range(length//(fps*interval)):
	trailer_cap.set(1, i*fps*interval);
	ret, frame = trailer_cap.read()
	if ret:
		features = desc.describe(frame[y:y+h,x:x+w])

		results=[(1,0)]
		print("Searching for frame "+str(i*fps*interval))
		if i>0:
			results = movie_searcher.search(features, skip_duration, mapping[i-1], mapping[i-1]+2*movie_searcher.getFPS()*interval)

		if results[0][0] > threshold:
			results = movie_searcher.search(features, skip_duration)
		else:
			print('hit')

		# if results[0][0] < threshold:
		mapping.append(results[0][1])
		print(results[0])
		cv2.imwrite("frame_"+str(i*fps*interval)+".jpg", frame[y:y+h,x:x+w])
		movie_cap.set(1, int(results[0][1]))
		ret, frame = movie_cap.read()
		cv2.imwrite("frame_"+str(i*fps*interval)+"_matched.jpg", frame)
		# else:
		# 	mapping.append(0)
		# 	print("No match found")

print(mapping)
