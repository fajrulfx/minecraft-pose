import cv2
import mediapipe as mp
import pydirectinput as pdi
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

# For webcam input:
squat = False
jump = False
click = False
walk = False

cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    if results.pose_landmarks:
        image_height, image_width, _ = image.shape
        
        elbow_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y) * image_height/2
        wrist_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y) * image_height/2
        hip_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y) * image_height/2
        normalfactor = abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x - results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x)
        
        # ketika jongkok, pencet shift
        # jongkok dideteksi ketika posisi kepal tangan ada di bawah pinggul
        if wrist_y > hip_y and squat==False and walk==False:
          squat = True
          pdi.keyDown('shift')

        elif wrist_y < hip_y and squat==True:
          squat = False
          pdi.keyUp('shift')

        # ketika loncat, pencet spasi
        # loncat dideteksi ketika posisi kepal tangan berada di atas sikut
        if elbow_y > wrist_y and jump==False and results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y:
          jump = True
          pdi.press('space')

        elif elbow_y < wrist_y and jump==True:
          jump = False

        # deteksi klik
        # tangan kanan di atas dan kiri di bawah, lalu tangan kanan turun
        if results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y and results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y > results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y:
          click = True

        elif results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y > results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y and click == True:
          click = False
          pdi.click()

        # kaki naik, pencet w
        if results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y < (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y)/2 and walk == False:
          walk = True
          pdi.keyDown('w')
          pdi.keyUp('shift')
        elif results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y > (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y)/2 and walk == True:
          walk = False
          pdi.keyUp('w')
        
        # tangan kanan lurus, mouse gerak ke kanan
        if results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y < (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + normalfactor) and results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y > (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y - normalfactor):
          pdi.moveRel(100,0,2)
          #print('kanan')
        
        # tangan kiri lurus, mouse gerak ke kiri
        if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y < (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + normalfactor) and results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y > (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y - normalfactor):
          pdi.moveRel(-100,0,2)
          #print('kiri')


    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
