# import the necessary packages
import argparse
import pickle
from pathlib import Path
import cv2
from searcher import Searcher
from rgbhistogram import RGBHistogram
from indexmovie import index_movie

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,	help="Path to the image")
ap.add_argument("-m", "--movie", required=True, help="Path to the movie")
ap.add_argument("-s", "--skip", required=True, help="Duration to skip checking")
args = vars(ap.parse_args())

movie_index_filename = args["movie"].rsplit(".", 1)[0]+'.db'
index_file = Path(movie_index_filename)
if index_file.is_file():
	print("Loading movie index from "+movie_index_filename)
	movie_index = pickle.load(open(movie_index_filename, 'rb'))
else:
	print("Index file not found, indexing movie...")
	movie_index = index_movie(args["movie"], movie_index_filename)

print("Done")

movie_searcher = Searcher(movie_index)
# results = searcher.search(queryFeatures)

image = cv2.imread(args["image"])
movie_cap = cv2.VideoCapture(args["movie"])

desc = RGBHistogram([8, 8, 8])
skip_duration = float(args["skip"])

features = desc.describe(image)
results = movie_searcher.search(features, skip_duration)

movie_cap.set(1, int(results[0][1]))
ret, frame = movie_cap.read()
cv2.imwrite("result.jpg", frame)

print(results[0])
