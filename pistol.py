import os
import cv2
import numpy as np
import threading
from datetime import datetime, timedelta
from playsound import playsound
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from twilio.rest import Client
from dotenv import load_dotenv
load_dotenv()

# Load environment variables
twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')

# Twilio phone numbers
sender_number = os.getenv('sender_number')
recipient_number = os.getenv('recipient_number')

# Google credentials
credentials_path = r"C:\Users\Nithesh\Desktop\Hackprix\android-eye-crime-detectiom-96e66e0abcd6.json"

# Flags and state
pistol_detected = False
recording = False
recorded_frames = []

# Thresholds (ensure you set these in your actual code)
confThreshold = 0.5
nmsThreshold = 0.4
classNames = ["Pistol"]  # Include more classes as needed

# Pistol detection and bounding box drawing
def findObjects(outputs, img):
    global pistol_detected, recording, recorded_frames
    hT, wT, _ = img.shape
    boundingBoxes = []
    classIndexes = []
    confidenceValues = []

    for output in outputs:
        for detection in output:
            probScores = detection[5:]
            classIndex = np.argmax(probScores)
            confidence = probScores[classIndex]

            if confidence >= confThreshold and classNames[classIndex] == "Pistol":
                if not pistol_detected:
                    pistol_detected = True
                    recording = True
                    print("Pistol detected")
                    threading.Thread(target=playsound, args=(r"C:\Users\Nithesh\Desktop\Hackprix\emergency-alarm-with-reverb-29431.mp3",)).start()

                w, h = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int((detection[0] * wT) - w / 2), int((detection[1] * hT) - h / 2)

                boundingBoxes.append([x, y, w, h])
                classIndexes.append(classIndex)
                confidenceValues.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boundingBoxes, confidenceValues, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        box = boundingBoxes[i]
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  
        cv2.rectangle(img, (x - 1, y - 25), (x + w + 1, y), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "weapon detected - Pistol", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
