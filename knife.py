import cv2
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from twilio.rest import Client
from datetime import datetime, timedelta
from playsound import playsound
import threading

# Alert sound function


def play_alert_sound():
    playsound(r"C:\Users\Nithesh\Desktop\start\Crime-Detection-Model-main\emergency-alarm-with-reverb-29431.mp3")


# Initialize Google Drive service


credentials_path = r"C:\Users\Nithesh\Desktop\start\Crime-Detection-Model-main\android-eye-crime-detectiom-96e66e0abcd6.json"
credentials = service_account.Credentials.from_service_account_file(credentials_path)
drive_service = build('drive', 'v3', credentials=credentials)

# Initialize the Twilio client

twilio_account_sid = 'AC3038d75fba26755c15e0e76206550ef4'
twilio_auth_token = '1331048c70f9c059db76ad778aa83899 '
twilio_client = Client(twilio_account_sid, twilio_auth_token)

# Twilio configurations

sender_number = '+12723185435'  # Twilio sandbox number
recipient_number = '+919342250779'

# Paths to YOLOv3 files and COCO class names

weights_path = r"C:\Users\Nithesh\Desktop\start\Crime-Detection-Model-main\yolov3.weights"
config_path = r"C:\Users\Nithesh\Desktop\start\Crime-Detection-Model-main\yolov3.cfg"
names_path = r"C:\Users\Nithesh\Desktop\start\Crime-Detection-Model-main\coco.names.txt"



# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()

# Load COCO class labels

with open(names_path, 'r') as f:
    classes = f.read().strip().split('\n')

# Define classes of interest (e.g., knife)

classes_of_interest = ["knife"]

# Open the default camera

cap = cv2.VideoCapture(0)  # 0 for the default camera, you may need to change this number if you have multiple cameras

# Set the resolution of the video capture object

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Height

# Flag to keep track of knife detection

knife_detected = False

# Initialize variables for recording

recording = False
recorded_frames = []
start_recording_time = None

while True:
    # Capture frame-by-frame

    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally

    frame = cv2.flip(frame, 1)

    height, width, channels = frame.shape

    # Preprocess image

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Get the indices of the output layers

    output_layer_indices = net.getUnconnectedOutLayers()

    # Map the indices to the corresponding layer names

    output_layer_names = [layer_names[i - 1] for i in list(output_layer_indices)]

    # Perform forward pass through the network

    outs = net.forward(output_layer_names)

    # Process detection results

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                detected_class = classes[class_id]
                # Check if the detected class is one of the classes of interest

                if detected_class in classes_of_interest:
                    if not knife_detected:
                        print("Crime Alert: Weapon detected. Sending crime alert to headquarters with sample video URL.")
                        knife_detected = True  # Set the flag to True to indicate knife detection
                        # Start recording

                        recording = True
                        start_recording_time = datetime.now()
                        # Alert sound

                        alert_sound = threading.Thread(target=play_alert_sound)
                        alert_sound.start()

                # Object detected

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y + 30), font, 3, color, 3)

    # Display the output

    cv2.imshow("Object Detection", frame)

    # Check for the 'q' key pressed to exit

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # If recording, save frames

    if recording:
        recorded_frames.append(frame)

    # If recording time exceeds 5 seconds, stop recording

    if recording and (datetime.now() - start_recording_time) >= timedelta(seconds=5):
        # Save recorded frames to video file

        out = cv2.VideoWriter('recorded_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame.shape[1], frame.shape[0]))
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

        message_body = f'Crime detected! Please confirm if it is a crime by checking the below video (location- v laptop cam): {file_url}'
        twilio_client.messages.create(from_=sender_number, body=message_body, to=recipient_number)
        print("WhatsApp message sent with video URL.")

        # Reset variables

        recording = False
        recorded_frames = []

# Release the capture and close all windows

cap.release()
cv2.destroyAllWindows()