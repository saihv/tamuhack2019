import cv2
import cognitive_face as CF
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Define the following for the rest of the functions
APIKey = 'db9b516ba0414434a511351345550f91' # make sure to fill in the key you obtained for Face API
BaseURL = 'https://southcentralus.api.cognitive.microsoft.com/face/v1.0'  # Replace with your regional Base URL

PERSON_GROUP_ID = 'auth-person'

def setupFaceAPI():
    CF.Key.set(APIKey)
    CF.BaseUrl.set(BaseURL)

def captureImage(file_name):
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

def matchFaces(faces):
    similarity = CF.face.verify(faces[0][0]['faceId'], faces[1][0]['faceId'])
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

def verifyImage():
    setupFaceAPI()
    filename = 'face_pic.png'
    captureImage(filename)
    faces = detectFaces(filename)
    identifiedFaces = identifyFaces(faces)
    imageAuthStatus, conf = findAuthFace(identifiedFaces, faces, filename)
    return imageAuthStatus, conf