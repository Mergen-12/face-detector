import cv2
import dlib

# Load face detector (Haar Cascade) and facial landmark predictor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor_path = 'C:\\Users\\dvt\\Dev\\Data Science and Machine Learning\\mediacloak_patch_01\\models\\shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

# Load input video
video_path = 'C:\\Users\\dvt\\Dev\\Data Science and Machine Learning\\Testing\\TestSet01.mp4'
cap = cv2.VideoCapture(video_path)

# Define output video codec and create VideoWriter object
output_video_path = 'output.mp4'

# Define output video codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 format
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Convert face region to dlib rectangle
        rect = dlib.rectangle(x, y, x + w, y + h)
        
        # Detect facial landmarks within the face region
        landmarks = predictor(gray, rect)
        
        # Convert dlib landmarks to a list of (x, y) coordinates
        landmarks_points = [(p.x, p.y) for p in landmarks.parts()]
        
        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw facial landmarks on the frame
        for (x, y) in landmarks_points:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    
    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('Video', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
