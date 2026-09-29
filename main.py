import cv2
import mediapipe as mp 


cap =cv2.VideoCapture(0)
cap.set(3,1280) # width
cap.set(4,720) # height 




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
 


 




