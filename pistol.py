from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
credentials_path = r"C:\Users\Nithesh\Desktop\Hackprix\android-eye-crime-detectiom-96e66e0abcd6.json"
import cv2
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from twilio.rest import Client
from datetime import datetime, timedelta
import threading
from playsound import playsound  # Import the playsound library for playing the alert sound
# Flag to track if pistol has been detected before
pistol_detected = False
recording = False
recorded_frames = []

# Function to find objects and draw bounding boxes
def findObjects(outputs, img):
    global pistol_detected, recording, recorded_frames
    hT, wT, cT = img.shape
    boundingBoxes = []
    classIndexes = []
    confidenceValues = []

    for output in outputs:
        for detection in output:
            probScores = detection[5:]
            classIndex = np.argmax(probScores)
            confidence = probScores[classIndex]

            if confidence >= confThreshold and classNames[classIndex] == "Pistol":
                if not pistol_detected:  # Print message only if pistol has not been detected before
                    pistol_detected = True
                    print("Pistol detected")
                    recording = True
                    # Play the alert sound
                    threading.Thread(target=playsound, args=(r"C:\Users\Nithesh\Desktop\Hackprix\emergency-alarm-with-reverb-29431.mp3",)).start()

                w, h = int(detection[2] * wT), int(detection[3] * hT)
                x, y = int((detection[0] * wT) - w / 2), int((detection[1] * hT) - h / 2)

                boundingBoxes.append([x, y, w, h])
                classIndexes.append(classIndex)
                confidenceValues.append(float(confidence))

                indices = cv2.dnn.NMSBoxes(boundingBoxes, confidenceValues, confThreshold, nmsThreshold)

                for i in indices:
                    box = boundingBoxes[i]
                    x, y, w, h = box[0], box[1], box[2], box[3]

                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(img, (x - 1, y - 25), (x + w + 1, y), (0, 255, 0), cv2.FILLED)

                    # Display the output message for pistol detection
                    cv2.putText(img, "weapon detected - Pistol",
                                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)