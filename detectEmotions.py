import cv2
import cognitive_face as CF
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import time

# Define the following for the rest of the functions
face_key = '12d9fd6f490c4dcaa84bfdd8c3a06126'
face_base_url = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0'
CF.Key.set(face_key)
CF.BaseUrl.set(face_base_url)

vision_base_url = "https://westcentralus.api.cognitive.microsoft.com/vision/v2.0/"

analyze_url = vision_base_url + "analyze"
vision_key = '69258ea2b86a46bdbdc635b11eed31d6'


def img_capture(file_name):
	cam = cv2.VideoCapture(0)
	
	cv2. namedWindow('WebCam')
	
	img_counter = 0
	
	while True:
		ret, frame = cam.read()
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
	
def img_cap_batch(base_file_name, quantity):
	counter = 0
	try:
		while counter < quantity:
			file_name = base_file_name + str(counter)
			img_capture(file_name)
			counter += 1
		
	except KeyboardInterrupt:
		print('Keyboard interrupt activated, stopping image capture')
	return 0
	
def local_img_creation(file_name, analyze_url, vision_key):
	image_data = open(file_name, "rb").read()
	headers    = {'Ocp-Apim-Subscription-Key': vision_key,'Content-Type': 'application/octet-stream'}
	params     = {'visualFeatures': 'Categories,Description,Color'}
	response = requests.post(analyze_url, headers=headers, params=params, data=image_data)
	response.raise_for_status()
	
	# The 'analysis' object contains various fields that describe the image. The most
	# relevant caption for the image is obtained from the 'description' property.
	analysis = response.json()
	print(analysis)
	image_caption = analysis["description"]["captions"][0]["text"].capitalize()
	
	# Display the image and overlay it with the caption.
	image = Image.open(BytesIO(image_data))
	return image
	
def face_det(file_name):
	faces = CF.face.detect(file_name)
	return faces

def face_match(faces):
	similarity = CF.face.verify(faces[0][0]['faceId'], faces[1][0]['faceId'])
	return similarity
def findDetectionRect(face):
	rect = face['faceRectangle']
	left = rect['left']
	top = rect['top']
	bottom = top + rect['height']
	right = left + rect['width']
	return (left, top), (right, bottom)

def max_emotion(face):
	fa = face['faceAttributes']
        ems = fa['emotion']
	max_em = None
	max_em_val = None			
	for em in ems:
		if max_em == None:
			max_em = em
			max_em_val = ems[em]
		elif max_em_val <= ems[em]:
			max_em_val = ems[em]
			max_em = em
	return max_em, max_em_val	


def face_feed(file_name):
	#cam = cv2.VideoCapture(0)
	
	cv2.namedWindow('WebCam')
	
	img_counter = 0
	
	while True:
		cam = cv2.VideoCapture(0)
		ret, frame = cam.read()

		#cv2.imshow('WebCam', frame)
		
		if not ret:
			break
		k = cv2.waitKey(1)
		img_name = file_name
		cv2.imwrite(img_name, frame)
		print("{} written!".format(img_name))
		if k%256 == 27:
			# ESC Pressed
			print("Escape hit, closing")
			break
			cam.release()
		faces = CF.face.detect(img_name, attributes='emotion')
		for face in faces:
			fa = face['faceAttributes']
        		#hp = fa['headPose']
			ems = fa['emotion']
            
			#print(hp['roll'])
			print(ems)
			pt1, pt2 = findDetectionRect(face)
			print(pt1)
			print(pt2)
			cv2.rectangle(frame, pt1, pt2, (0,0,255), 5)
			max_em = None
			max_em_val = None			
			for em in ems:
				if max_em == None:
					max_em = em
					max_em_val = ems[em]
				elif max_em_val <= ems[em]:
					max_em_val = ems[em]
					max_em = em
					
			emString = str(max_em)
			cv2.putText(frame, emString, pt1, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
		cv2.imshow('WebCam',frame)
		cv2.waitKey(1)	
		time.sleep(2.5)
		cam.release()
	
	cv2.destroyAllWindows()
	return 0
if __name__ == '__main__':
	face_feed('test_img.png')
	
