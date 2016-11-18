# import the necessary packages
import cv2

def find_box(input_file):
	MAX_BLACK_CHECK = 5

	cap = cv2.VideoCapture(input_file)
	fps = round(cap.get(cv2.CAP_PROP_FPS))
	x = []
	y = []
	w = []
	h = []

	for i in range(5, 5+MAX_BLACK_CHECK):
		cap.set(1, i*fps)
		_, frame = cap.read(1)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		_, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
		points = cv2.findNonZero(thresh)

		x1, y1, w1, h1 = cv2.boundingRect(points)
		if x1+y1+w1+h1 > 0:
			x.append(x1)
			y.append(y1)
			w.append(w1)
			h.append(h1)

	return (min(x), min(y), max(w), max(h))
