import cv2
import tensorflow as tf
import numpy as np
import pickle
import sqlite3

model = tf.keras.models.load_model('trainedModel.h5')
#estimate = tf.keras.estimator.model_to_estimator(keras_model = model, model_dir = r'C:\Users\bhuta\Desktop\sl\sign-language-new')
prediction = None

def get_size():
	img = cv2.imread('gestures/1/100.jpg', 0)
	return img.shape

image_x, image_y = get_size()



def process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype = np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def probability(model, image):
	processed = process_image(image)
	pred_probability = model.predict(processed)[0]
	pred_class = list(pred_probability).index(max(pred_probability))
	return max(pred_probability), pred_class

def class_from_db(pred_class):
	conn = sqlite3.connect('gesture_db.db')
	cmd = "SELECT g_name FROM gesture WHERE g_id = " + str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

def split_to_array(text, num_of_words):
	list_words = text.split(" ")
	length = len(list_words)
	arr = []
	start = 0
	end = num_of_words
	while length > 0:
		word = ""
		for i in list_words[start : end]:
			word += " " + i
		arr.append(word)
		start += num_of_words
		end += num_of_words
		length -= num_of_words
	return arr

def print_text(board, arr):
	y = 200
	for word in arr:
		cv2.putText(board, word, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
		y += 25

def grid():
	with open("grid", "rb") as f:
		grid = pickle.load(f)
	return grid

def predictor():
	global prediction
	cam = cv2.VideoCapture(1)
	if cam.read()[0] == False:
		cam = cv2.VideoCapture(0)
	hist = grid()
	x, y, w, h = 300, 100, 300, 300
	while True:
		text = ""
		img = cam.read()[1]
		img = cv2.flip(img, 1)
		img = cv2.resize(img, (640, 480))
		imgCrop = img[y : y + h, x : x + w]
		imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
		disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10, 10))
		dst = cv2.filter2D(dst,-1,disc,dst)
		blur = cv2.GaussianBlur(dst, (11,11), 0)
		blur = cv2.medianBlur(blur, 15)
		#blur = cv2.bilateralFilter(blur, 9, 75, 75)
		thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
		thresh = cv2.merge((thresh,thresh,thresh))
		thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
		thresh = thresh[y : y + h, x : x + w]
#		kernel = np.ones((3, 3), np.uint8)
#		opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

#		#sure background area
#		sureBackground = cv2.dilate(opening, kernel, iterations = 3)

#		#sure foreground area
#		distanceTransform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
#		_, sureForeground = cv2.threshold(distanceTransform, 0.7 * distanceTransform.max(), 255, 0)

#		#unknown region
#		sureForeground = np.uint8(sureForeground)
#		unknown = cv2.subtract(sureBackground, sureForeground)
#		known = cv2.subtract(sureForeground, sureBackground)

		contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			#print(cv2.contourArea(contour))
			if cv2.contourArea(contour) > 10000:
				x1, y1, w1, h1 = cv2.boundingRect(contour)
				save_img = thresh[y1 : y1 + h1, x1 : x1 + w1]
				
				if w1 > h1:
					save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2) , int((w1 - h1) / 2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
				elif h1 > w1:
					save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2) , int((h1 - w1) / 2) , cv2.BORDER_CONSTANT, (0, 0, 0))
				
				pred_probability, pred_class = probability(model, save_img)
				
				if pred_probability * 100 > 80:
					text = class_from_db(pred_class)
					print(text)
		board = np.zeros((480, 640, 3), dtype = np.uint8)
		arr = split_to_array(text, 2)
		print_text(board, arr)
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
		res = np.hstack((img, board))
		cv2.imshow("recognizing gesture...", res)
		cv2.imshow("thresh", thresh)
		if cv2.waitKey(1) == ord('q'):
			break

probability(model, np.zeros((50, 50), dtype = np.uint8))
predictor()
