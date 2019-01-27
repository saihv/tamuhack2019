import cv2
import cognitive_face as CF
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# Define the following for the rest of the functions
APIKey = 'db9b516ba0414434a511351345550f91' # make sure to fill in the key you obtained for Face API
BaseURL = 'https://southcentralus.api.cognitive.microsoft.com/face/v1.0'  # Replace with your regional Base URL

PERSON_GROUP_ID = 'auth-person'

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 3



# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

authenticated = True

def setupFaceAPI():
    CF.Key.set(APIKey)
    CF.BaseUrl.set(BaseURL)

def detectBlinks():
    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0
    while True:
        if fileStream and not vs.more():
	        break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
        frame = vs.read()
        frameOrig = frame
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	    # detect faces in the grayscale frame
        rects = detector(gray, 0)

	    # loop over the face detections
        for rect in rects:
    		# determine the facial landmarks for the face region, then
		    # convert the facial landmark (x, y)-coordinates to a NumPy
		    # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

		    # extract the left and right eye coordinates, then use the
		    # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

		    # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

		    # compute the convex hull for the left and right eye, then
		    # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		    # check to see if the eye aspect ratio is below the blink
		    # threshold, and if so, increment the blink frame counter
            if ear < EYE_AR_THRESH:
                COUNTER += 1

		    # otherwise, the eye aspect ratio is not below the blink
		    # threshold
            else:
    			# if the eyes were closed for a sufficient number of
			    # then increment the total number of blinks
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1

			    # reset the eye frame counter
                COUNTER = 0

		    # draw the total number of blinks on the frame along with
		    # the computed eye aspect ratio for the frame
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    	# show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if TOTAL > 5:
            detectedBlinks = True
            break
    
    	# if the `q` key was pressed, break from the loop   
        if key == ord("q"):
            break

    vs.stop()
    cv2.destroyAllWindows()
    return detectedBlinks, frame
    

def captureImage(file_name):
    print('Opening video stream')
    cam = cv2.VideoCapture(0)    
    cv2.namedWindow('WebCam')    
    img_counter = 0
    
    while True:
        ret, frame = cam.read()
        print('Opened video stream')
        cv2.imshow('WebCam', frame)
        
        if not ret:
            break
        k = cv2.waitKey(1)
        
        if k%256 == 27:
            # ESC Pressed
            print("Escape hit, closing")
            break
        elif k%256 == 32:
            # Space pressed
            img_name = file_name
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            break
    cam.release()
    cv2.destroyAllWindows()
    return 0
    
def captureImageBatch(base_file_name, quantity):
    counter = 0
    try:
        while counter < quantity:
            file_name = base_file_name + str(counter)
            img_capture(file_name)
            counter += 1
        
    except KeyboardInterrupt:
        print('Keyboard interrupt activated, stopping image capture')
    return 0

def findDetectionRect(face):
    rect = face['faceRectangle']
    left = rect['left']
    top = rect['top']
    bottom = top + rect['height']
    right = left + rect['width']
    return (left, top), (right, bottom)
    
def detectFaces(imgFilename):
    faces = CF.face.detect(imgFilename)
    return faces

def matchFaces(blinkFaces, faces):
    similarity = CF.face.verify(blinkFaces[0][0]['faceId'], faces[0][0]['faceId'])
    return similarity

def identifyFaces(faces):
    face_ids = [f['faceId'] for f in faces]
    #print(face_ids)
    identified_faces = CF.face.identify(face_ids, PERSON_GROUP_ID)
    return identified_faces
    #print(identified_faces)

def findAuthFace(identified_faces, faces, imgFilename):
    authorized = False
    image = cv2.imread(imgFilename)
    conf = 0.0
    for i in identified_faces:
        print(i)
        if i['candidates']:
            id = i['candidates'][0]['personId']
            conf = i['candidates'][0]['confidence']
            faceId = i['faceId']
            name = CF.person.get(PERSON_GROUP_ID, id)

            for face in faces:
                if face['faceId'] == faceId:
                    pt1, pt2 = findDetectionRect(face)
                    #print(pt1)
                    #print(pt2)
                    cv2.rectangle(image, pt1, pt2, (0,0,255), 5)
                    cv2.putText(image, str(name['name']), (pt1[0], pt1[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                    cv2.putText(image, str(conf*100) + '%', pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                    authorized = True

    cv2.namedWindow('Detections')
    cv2.imshow('Detections', image)
    cv2.waitKey(0)

    return authorized, conf

if __name__ == '__main__':
    setupFaceAPI()
    filename = 'face_pic.png'
    blinksDetected, detectedFace = detectBlinks()
    blinkFace = 'blinkFace.png'
    cv2.imwrite(blinkFace, detectedFace)
    faces = detectFaces(blinkFace)

    #captureImage(filename)

    #faces = detectFaces(filename)
    identifiedFaces = identifyFaces(faces)
    imageAuthStatus, conf = findAuthFace(identifiedFaces, faces, blinkFace)

    #similarity = matchFaces(blinkFaces, faces)
    #print(similarity)
