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

    #convert BGR image to RGB 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
       for face in results.multi_face_landmarks:
         left_eye = face.landmark[468] 
         right_eye = face.landmark[473] 
         h, w = frame.shape[:2] #get frame height and width in pixels 
         for eye in (left_eye, right_eye):
            cv2.circle(frame, (int(eye.x * w), int(eye.y *h))), 3, (0,255,0), -1 # draw a green dot on iris


    





  
    
         #show cam
    cv2.imshow("Webcam", frame)

    # press q to exit 
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break
 


 




