
from pyparsing import empty
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
from mylib import config, thread, detectify,speed_prediction
#import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest
import math
import cv2
import sys, os
import numpy as np
from threading import Thread
from queue import Queue
import io
import imutils
import easyocr
import pandas as pd




def run():
	totalFrames=0
	a=detectify.EuclideanDistTracker()
	ROI=250
	savedid=[]
	prevtime=0
	newtime=0
	df4 = pd.DataFrame(columns= ["TrackingID","Speed",'Image','Frame_number'])
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=False,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model",required=True,
		help="path to Caffe pre-trained model")
	ap.add_argument("-i", "--input", default='',type=str,
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	# confidence default 0.4
	ap.add_argument("-c", "--confidence", type=float, default=0.4,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=30,
		help="# of skip frames between detections")
	args = vars(ap.parse_args())

	# initialize the list of class labels MobileNet SSD was trained to
	# detect
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	# load our serialized model from disk
	net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])

	# if a video path was not supplied, grab a reference to the ip camera
	if not args.get("input", False):
		print("[INFO] Starting the live stream..")
		vs = VideoStream(config.url).start()
		time.sleep(2.0)

	# otherwise, grab a reference to the video file
	else:
		print("[INFO] Starting the video..")
		vs = cv2.VideoCapture(args["input"])

	# initialize the video writer (we'll instantiate later if need be)
	writer = None

	# initialize the frame dimensions (we'll set them as soon as we read
	# the first frame from the video)
	W = None
	H = None
	

	# instantiate our centroid tracker, then initialize a list to store
	# each of our dlib correlation trackers, followed by a dictionary to
	# map each unique object ID to a TrackableObject
	

	# initialize the total number of frames processed thus far, along
	# with the total number of objects that have moved either up or down
	



	# start the frames per second throughput estimator

	if config.Thread:
		vs = thread.ThreadingClass(config.url)
	
		# loop over frames from the video stream
	while True:
		
		# grab the next frame and handle if we are reading from either
		# VideoCapture or VideoStream
		frame = vs.read()
		if not frame[0]:
			break
		frame = frame[1] if args.get("input", False) else frame
		cv2.line(frame,(350,700),(952,700),(0,255,0),1)
		
		# if we are viewing a video and we did not grab a frame then we
		# have reached the end of the video
		if args["input"] is not None and frame is None:
			break

		newtime=time.time()
		# resize the frame to have a maximum width of 500 pixels (the
		# less data we have, the faster we can process it), then convert
		# the frame from BGR to RGB for dlib
		#frame = imutils.resize(frame, width = 500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		
		# if the frame dimensions are empty, set them
		if W is None or H is None:
			(H, W) = frame.shape[:2]

		# if we are supposed to be writing a video to disk, initialize
		# the writer
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)
		rects = []
			# set the status and initialize our new set of object trackers
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		net.setInput(blob)
		detections = net.forward()
		for i in np.arange(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > 0.4:
				idx = int(detections[0, 0, i, 1])
				# if the class label is not a car, ignore it
				if (CLASSES[idx] == "car" or CLASSES[idx] == "motorbike" or CLASSES[idx]=="bus" or CLASSES[idx]=="train"):
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")
					rects.append([startX, startY, endX, endY])
		if(rects):
			boxes_ids = a.update(rects)
			for box_id in boxes_ids:
						x,y,w,h,id = box_id
						crop=frame[y:h,x:w]
						if(y>600 and id not in savedid and y<700):
							file = 'SavedVehicles//' + str(totalFrames) + '.jpg'
							cv2.imwrite(file,crop)
							savedid.append(id)
						speed=0
						if(y>250):
							speed=speed_prediction.predict_speed(y,totalFrames,ROI)
						
						cv2.putText(frame,str(speed)+" Km/hr",(x,y-15), cv2.FONT_HERSHEY_PLAIN,1,(255,255,0),2)
						#df3 = pd.DataFrame([[id,speed,crop,totalFrames]], columns= ["TrackingID","Speed",'Image','Frame_number'])
						#df4=df4.append(df3,ignore_index=True)

						
						cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 1)
						
						

					#DISPLAY
					#cv2.imshow("Mask",mask2)
					#cv2.imshow("Erode", e_img)
		cv2.imshow("ROI", frame)
		fps = 1/(newtime-prevtime)
		prevtime = newtime
		#print(int(fps))

		if cv2.waitKey(1) & 0xff==ord('q'):
			break
		totalFrames=totalFrames+1

	vs.release()
cv2.destroyAllWindows()

run()