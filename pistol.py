import cv2
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from twilio.rest import Client
from datetime import datetime, timedelta
import threading
from playsound import playsound  # Import the playsound library for playing the alert sound

# Video capture from webcam
cap = cv2.VideoCapture(0)

# Width and Height of network's input image
whT = 320
# Object detection confidence threshold
confThreshold = 0.5
# Non-maximum Suppression threshold
nmsThreshold = 0.3

# Class names
classNames = ["Pistol"]

# Model paths
modelConfiguration = r"C:\Users\Nithesh\Desktop\Hackprix\yolov3-custom.cfg"
modelWeights = r"C:\Users\Nithesh\Desktop\Hackprix\yolov3-custom_2000.weights"

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Google Drive credentials path
credentials_path = r"C:\Users\Nithesh\Desktop\Hackprix\android-eye-crime-detectiom-96e66e0abcd6.json"

# Twilio credentials
twilio_account_sid = my_sid #secret sid in string should be given
twilio_auth_token = my_auth_token #secret token in string should be given
sender_number = sender_no  # Twilio sandbox number in string should be given
recipient_number = recipient_no  # Replace with the actual recipient's number in string

# Initialize Google Drive service
credentials = service_account.Credentials.from_service_account_file(credentials_path)
drive_service = build('drive', 'v3', credentials=credentials)

# Initialize Twilio client
twilio_client = Client(twilio_account_sid, twilio_auth_token)

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

# Function to start recording
def start_recording():
    global recording, recorded_frames
    while recording:
        success, frame = cap.read()
        if success:
            recorded_frames.append(frame)

# Process frames from webcam
while cap.isOpened():
    success, frame = cap.read()

    if success:
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], swapRB=True, crop=False)
        net.setInput(blob)
        outputLayersNames = net.getUnconnectedOutLayersNames()
        outputs = net.forward(outputLayersNames)
        findObjects(outputs, frame)

        # Display video feed
        cv2.imshow('Pistol Detection', frame)

        if recording:
            recording_thread = threading.Thread(target=start_recording)
            recording_thread.start()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            recording = False
            recording_thread.join()  # Wait for recording thread to finish
            break

    else:
        print("Failed to read frame")

# Save recorded frames to video file
if recorded_frames:
    out = cv2.VideoWriter('recorded_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30,
                          (recorded_frames[0].shape[1], recorded_frames[0].shape[0]))
    for frame in recorded_frames:
        out.write(frame)
    out.release()

    # Upload video to Google Drive
    file_metadata = {'name': 'recorded_video.avi'}
    media = MediaFileUpload('recorded_video.avi', mimetype='video/avi')
    uploaded_file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = uploaded_file.get('id')

    # Set permissions for the uploaded video
    drive_service.permissions().create(
        fileId=file_id,
        body={'role': 'reader', 'type': 'anyone'},
        fields='id'
    ).execute()

    # Generate public URL for the uploaded video file
    file_url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"

    # Send WhatsApp message with Google Drive public URL
    message_body = f'Pistol detected! View the recorded video here: {file_url}'
    twilio_client.messages.create(from_=sender_number, body=message_body, to=recipient_number)
    print("WhatsApp message sent to headquarters with video URL")

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
