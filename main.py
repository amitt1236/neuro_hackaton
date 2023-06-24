import mediapipe as mp
import cv2
import gaze
import helpers
import game 
import sys
import os
from pathlib import Path
import numpy as np

# root path
path = Path(__file__)
main_path = path.parent.absolute()
sys.path.append(os.path.join(main_path,'stress/'))

from stress.neurofeedback import READ_EEG

mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model
click = 0
calibration = [[],[]]

'''
Blink 
'''
blink_flag = True               # blinking flag, true if eyes are closed
blink_counter = 0               # counts the number of blinks from the beginning of the stream
BLINK_THRESHOLD = 20          # threshold to consider a blink

# from multiprocessing.pool import ThreadPool
# ee = READ_EEG()
# eeg_reader = ee.get_sampl
# pool = ThreadPool(processes=1)
stress_val = 0
stress_val_c = 0

# camera stream:
randerer_gui = game.renderer()
cap = cv2.VideoCapture(0)  # chose camera index (try 1, 2, 3)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # number of faces to track in each frame
        refine_landmarks=True,  # includes iris landmarks in the face mesh model
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        # async_result = pool.apply_async(eeg_reader)

        success, image = cap.read()
        if not success:  # no frame input
            print("Ignoring empty camera frame.") 
            continue
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # frame to RGB for the face-mesh model
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # frame back to BGR for OpenCV

        if results.multi_face_landmarks:
            x, y = gaze.gaze(image, results.multi_face_landmarks[0], calibration)  # gaze estimation
            
            # return_val = async_result
            # stress_val = return_val.get()
            stress_val_c = stress_val_c + 0.02 + np.random.normal(0, 0.05)
            stress_val = 7 + (np.cos(stress_val_c))
            print(stress_val)
            # stress_val = 0 if stress_val < 0 else 1 if stress_val > 1 else stress_val
            # stress_val = stress_val * 10

            if  x > 150 and click == 0: 
                print('click right')
                click = 1
                randerer_gui.render_gui(gaze_input=2, stress_level=stress_val)
            if  x < -150 and click == 0:
                print('click left')
                click = 2
                randerer_gui.render_gui(gaze_input=1, stress_level=stress_val)
            if abs(x) < 100:
                click = 0

            # blink detection
            blink_val = helpers.eyes_close(results.multi_face_landmarks[0])
            if blink_val < BLINK_THRESHOLD:
                blink_counter = blink_counter + 1
            if blink_val > BLINK_THRESHOLD * 2:
                blink_counter = 0
                blink_flag = False

            if blink_counter > 10 and not blink_flag:
                blink_flag = True
                blink_counter = 0
                randerer_gui.render_gui(gaze_input=3, stress_level=stress_val)

            randerer_gui.render_gui(gaze_input=0, stress_level=stress_val)


        cv2.imshow('output window', image)
        if cv2.waitKey(2) & 0xFF == 27:
            break
cap.release()
