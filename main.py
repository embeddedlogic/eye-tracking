import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)   # height

# initialize face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # puts dots on the eyes (10 extra dots)
    min_detection_confidence=0.5,  # how sure it must be to say "that's a face" (0 to 1)
    min_tracking_confidence=0.5,   # how sure it must be to keep following the face
)

while True:
    # read camera frames
    success, frame = cap.read()

    if not success:
        print("Not working loser.")
        continue

    # flip video 
    frame = cv2.flip(frame, 1)

    # get the frame's height and width in pixels 
    h, w = frame.shape[:2]

    # convert BGR image to RGB 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # only run this if a face was found (otherwise multi_face_landmarks is None)
    if results.multi_face_landmarks:
        # grab the first (and only) face
        face = results.multi_face_landmarks[0]

        # draw the iris centers (468 = right eye, 473 = left eye) in red
        for i in [468, 473]:
            # landmark x and y are between 0 and 1, so multiply by width/height to get pixels
            x = int(face.landmark[i].x * w)
            y = int(face.landmark[i].y * h)
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # BGR color, -1 = filled

        # draw eye corners and eyelids in green
        # 33, 133 = right eye corners | 159, 145 = right eye top/bottom lid
        # 362, 263 = left eye corners | 386, 374 = left eye top/bottom lid
        for i in [33, 133, 159, 145, 362, 263, 386, 374]:
            x = int(face.landmark[i].x * w)
            y = int(face.landmark[i].y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # show cam
    cv2.imshow("Webcam", frame)

    # press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera and close the window when done
cap.release()
cv2.destroyAllWindows()