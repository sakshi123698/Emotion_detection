import cv2
from keras.models import model_from_json
import numpy as np
import os  # For path verification

# File path to the model architecture JSON
#json_file_path =  r"D:\from scratch try\ emotiondetector.json"

json_file_path = r"D:\from scratch try\ emotiondetector.json"  # Use raw string notation

# Check if the file exists
if not os.path.isfile(json_file_path):
    print(f"File not found: {json_file_path}")
   
     # Optionally list directory contents
    directory = os.path.dirname(json_file_path)
    print("Files in directory:")
    print(os.listdir(directory))
    exit()

#print(type(model_json))  # Should be <class 'str'>

   

# Define the file paths
json_file_path = r"D:\from scratch try\ emotiondetector.json"
h5_file_path = r"D:\from scratch try\emotiondetector.h5"

# Check if the files exist
if not os.path.isfile(json_file_path):
    print(f"Error: JSON file not found at {json_file_path}")
    exit()
if not os.path.isfile(h5_file_path):
    print(f"Error: H5 file not found at {h5_file_path}")
    exit()

# Load model architecture from JSON file
with open(json_file_path, 'r') as json_file:
    model_json = json_file.read()

# Load the model from JSON
try:
    model = model_from_json(model_json)
except Exception as e:
    print(f"Error loading model from JSON: {e}")
    exit()

# Load model weights
try:
    model.load_weights(h5_file_path)
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit()

print(type(model_json))  # Should be <class 'str'>

# Load model architecture from JSON file
with open(json_file_path, "r") as json_file:
    model_json = json_file.read()



model = model_from_json(model_json)  # Load the model architecture from JSON
model.load_weights("emotiondetector.h5")  # Load the weights

# Load Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Labels corresponding to emotion classes
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Function to preprocess and normalize the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start video capture
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read frame from webcam
    ret, im = webcam.read()

    if not ret:
        print("Failed to capture image")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Loop through all detected faces
    for (x, y, w, h) in faces:
        # Extract face region from the gray image
        face = gray[y:y + h, x:x + w]

        # Draw rectangle around the face
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Resize the face to 48x48 for the model input
        face = cv2.resize(face, (48, 48))

        # Preprocess the face image
        img = extract_features(face)

        # Predict the emotion
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]

        # Display the prediction on the image
        cv2.putText(im, prediction_label, (x - 10, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

    # Show the frame with predictions
    cv2.imshow("Emotion Detection", im)

    # Exit on pressing 'Esc' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the webcam and close the window
webcam.release()
cv2.destroyAllWindows()
