from pydoc import cli
from tkinter import RIGHT
import cv2
import mediapipe as mp
import pydirectinput as pdi
import numpy as np
import time
import subprocess

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
spacehold = False
click = False
click_hold = False
right_click = False
walk = False
right = False
left = False
mouseup = False
mousedown = False

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
        hand_distance = abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x-results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x)

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
        if elbow_y > wrist_y and jump==False and results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y and results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y and hand_distance > 0.1:
          jump = True
          pdi.press('space')

        elif elbow_y < wrist_y and jump==True:
          jump = False

        # deteksi tangan sambung (control mouse atas bawah)
        if hand_distance < 0.06 and elbow_y > wrist_y and mouseup == False:
          mouseup = True
          moveup = subprocess.Popen(["python", "mouseup.py"], stdin=None, stdout=None, stderr=None, close_fds=True)  
        elif hand_distance < 0.06 and elbow_y < wrist_y and mousedown == False:
          mousedown = True
          movedown = subprocess.Popen(["python", "mousedown.py"], stdin=None, stdout=None, stderr=None, close_fds=True)
        elif hand_distance > 0.06 and (mouseup == True or mousedown == True):
          mouseup = False
          mousedown = False
          try:
            pollup = moveup.poll()
            if pollup is None:
              moveup.terminate()
            polldown = movedown.poll()
            if polldown is None:
              movedown.terminate()
          except:
            pass
        '''
        if hand_distance < 0.06 and elbow_y < wrist_y and mousedown == False:
          mousedown = True
          movedown = subprocess.Popen(["python", "mousedown.py"], stdin=None, stdout=None, stderr=None, close_fds=True)  
        elif hand_distance > 0.06 and mousedown == True:
          mouseup = False
          mousedown = False
          try:
            poll = moveup.poll()
            if poll is None:
              movedown.terminate()
          except:
            pass
        '''

        # deteksi klik
        # tangan kanan di atas dan kiri di bawah, lalu tangan kanan turun
        if results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y and results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y > results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y:
          click = True

        elif results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y > results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y and click == True:
          click = False
          if click_hold == False: 
            #pdi.leftClick()
            pdi.mouseDown()
            click_hold = True
          else:
            pdi.mouseUp()
            click_hold = False
        
        # deteksi klik kanan
        # tangan kiri di atas dan kanan di bawah, lalu tangan kiri turun
        # print('kiri:',results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x)
        # print('kanan:',results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x)
        # print('kanan:',abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x-results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x))
        
        if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y and results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y > results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y:
          right_click = True 

        elif results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y > results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y and right_click == True:
          right_click = False
          pdi.rightClick()
        
        # deteksi hold click
        # tangan kiri di atas dan kanan di bawah, lalu tangan kiri turun
        '''
        if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y and results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y > results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y:
          click = True

        elif results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y > results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y and click == True:
          click = False
          pdi.click(button=RIGHT)
        '''

        # kaki kanan naik, pencet w
        if results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y < (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y)/2 and walk == False:
          walk = True
          pdi.keyUp('shift')
          pdi.keyDown('w')
        elif results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y > (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y)/2 and walk == True:
          walk = False
          pdi.keyUp('w')
        
        # kaki kiri naik, pencet spasi hold
        if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y < (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y)/2 and spacehold == False:
          spacehold = True
          pdi.keyUp('shift')
          pdi.keyDown('space')
        elif results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y > (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y + results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y)/2 and spacehold == True:
          spacehold = False
          pdi.keyUp('space')
        
        # tangan kanan lurus, mouse gerak ke kanan
        if results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y < (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + normalfactor) and results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y > (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y - normalfactor) and right ==False:
          right = True
          moveright = subprocess.Popen(["python", "mouseright.py"], stdin=None, stdout=None, stderr=None, close_fds=True)
          #print('kanan')
        elif results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y > (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + normalfactor) and right == True:
          right = False
          try:
            poll = moveright.poll()
            if poll is None:
              moveright.terminate()
          except:
            pass
        
        # tangan kiri lurus, mouse gerak ke kiri
        if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y < (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + normalfactor) and results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y > (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y - normalfactor) and left == False:
          left = True
          moveleft = subprocess.Popen(["python", "mouseleft.py"], stdin=None, stdout=None, stderr=None, close_fds=True)
          #print('kiri')
        elif results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y > (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + normalfactor) and left == True:
          left = False
          try:
            poll = moveleft.poll()
            if poll is None:
              moveleft.terminate()
          except:
            pass

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
