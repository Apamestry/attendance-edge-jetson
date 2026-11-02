
# pip install cmake
# pip install face_recognition
# pip install opencv-python
# pip install numpy
import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime 

#video_capture = cv2.VideoCapture(0)
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )

video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

#load known faces

indramal_image = face_recognition.load_image_file("photos/indramal.jpg")
indramal_encoding = face_recognition.face_encodings(indramal_image)[0]

ANOSH_image = face_recognition.load_image_file("photos/ANOSH.jpg")
ANOSH_encoding = face_recognition.face_encodings(ANOSH_image)[0]

known_faces_encoding = [indramal_encoding,ANOSH_encoding]
known_faces_names = ["INDRAMAL "," ANOSH "]

#list of expected students
students = known_faces_names.copy()


face_location = []
face_encoding = []
face_names = []

#get the current date and time

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

 # recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces_encoding, face_encoding)
        name=""
        face_distance = face_recognition.face_distance(known_faces_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            name = known_faces_names[best_match_index]

      # Add the text if a person is present
        face_names.append(name)
        if name in known_faces_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOftext = (10,100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name+ "Present", bottomLeftCornerOftext, font, fontScale, fontColor, thickness, lineType)
       
            if name in students:
                students.remove(name)
                print(students)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])

    cv2.imshow("ATTENDANCE SYSTEM", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
    
video_capture.release()
cv2.destroyAllWindows()
f.close()

# To close the webcam window press "q"
# In order to run this program on devices....One need to install Visual Build Tools
