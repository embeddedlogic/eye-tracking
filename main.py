import cv2
import mediapipe as mp 


cap =cv2.VideoCapture(0)
cap.set(3,1280) # width
cap.set(4,720) # height 
#initialize face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, # puts dots on the eyes(10 extra dots)
    min_detection_confidence= 0.5, #how sure it must be to say " thats a face" (0 to 1)
    min_tracking_confidence = 0.5, # how sure it must be to keep following the face
)

while True:
    #Read camera frames 
    success, frame = cap.read()

    #converting the color of the video 
    if not success:
        print("Not working loser.")
        continue
    
    #flip video and change from BGR TO RGB
    frame = cv2.flip(frame,1)
    
    #show cam
    cv2.imshow("Webcam", frame)

    # press q to exit 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 


 




