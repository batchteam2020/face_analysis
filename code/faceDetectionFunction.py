import cv2
import face_recognition as fr
import numpy as np
import os
import face_recognition
from datetime import datetime




def face_detection (image_path):
    img=cv2.imread(image_path,1)
    img=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2))) #it is symetric
	
	#this line will be used for using the harcascaded classifier for WINDOWS users.
	
	
    #face_cascade = cv2.CascadeClassifier('C:\\Users\\power\\Anaconda3\\envs\\tensorflow_cpu\\Library\\etc\\haarcascades\\haarcascade_frontalface_default.xml')

	
	#for ubunto users the location of haarcascaded classifier will be diffrent.

   # face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

    #gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #faces=face_cascade.detectMultiScale(gray_img , scaleFactor=1.5,minNeighbors=5)

    for x,y,w,h in faces:
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face_crop =img[y:y+h, x:x+w]
    
    cv2.imshow('Face Detection',face_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

#this will be used if the image path in 'E:\\.... and so on'

#face_detection('E:\\face-detection\\President_Barack_Obama.jpg')

#face_detection('President_Barack_Obama.jpg')


def smile_sad_detection(image):

        # eyes and smile and face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

        face = face_cascade.detectMultiScale(gray,1.3 , 5)

        for (face_x , face_y , face_width , face_height) in face :

                cv2.rectangle(image, (face_x, face_y), (face_x + face_width, face_y + face_height), (0, 0, 255), 1)

                cv2.putText(image, "face", (face_x, face_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 0, 0))

                gray_region_of_interest = gray[face_y : face_y+face_height , face_x : face_x+face_width]

                color_region_of_interest = image[face_y: face_y+face_height, face_x: face_x+face_width]

                smile = smile_cascade.detectMultiScale(gray_region_of_interest , 6, 6)

                smile_score = smile_cascade.detectMultiScale3(gray_region_of_interest,scaleFactor=6,minNeighbors=6,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE,outputRejectLevels=True)[2]

                eyes = eye_cascade.detectMultiScale(gray_region_of_interest ,5, 5)
                
		

                for (smile_x , smile_y , smile_width , smile_height) in smile :

                     cv2.rectangle(color_region_of_interest, (smile_x, smile_y), (smile_x + smile_width, smile_y + smile_height), (0, 255, 0), 1)

                     cv2.putText(color_region_of_interest,"smile" , (smile_x , smile_y-5) , cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0))

                     cv2.putText(color_region_of_interest, "smile score : {}".format(smile_score), (smile_x + 216, smile_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 0, 0))

                for (eye_x, eye_y, eye_width, eye_height) in eyes:

                     cv2.rectangle(color_region_of_interest, (eye_x, eye_y), (eye_x + eye_width, eye_y + eye_height), (255, 0, 0), 1)

                     cv2.putText(color_region_of_interest,"eye" , (eye_x , eye_y-5) , cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0))
                     
                #For cropping image
                for x,y,w,h in face:
					 img=cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
					 face_crop =img[y:y+h, x:x+w]     

                print(smile_score)
                cv2.imshow( "window", face_crop)
                filename = datetime.now()
                cv2.imwrite("Output"+str(filename) + ".jpg", face_crop )
                
                cv2.waitKey(0)

#image = cv2.imread('messi2.jpg')
 
#output = smile_sad_detection(image)
#print(output)




"""
the used option is identification of facial recognition , it compare the input image to test with all images 
in faces folder it is like a database to train a Dlib
"""


def get_encoded_faces():

    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def unknown_image_encoded(img):
   
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def face_recognizer(im):
   
	#Recognizer of faces
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
  
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a label with a name below the face
            cv2.rectangle(img, (left-50, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left -20, bottom + 2), font, 1.0, (255, 255, 255), 2)


    # Display the resulting image
    while True:

        cv2.imshow('image', img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            return face_names 


print(face_recognizer("Output2019-09-20 23:42:51.811298.jpg"))



	














