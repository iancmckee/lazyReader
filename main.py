import cv2
import numpy as np

from pynput.keyboard import Key, Controller
from pynput import keyboard

keyboardController = Controller()


# dinstance function
def distance(x, y):
    import math
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


# capture source video
cap = cv2.VideoCapture(0)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# path to face cascde
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')


# function to get coordinates
def get_coords(p1):
    try:
        return int(p1[0][0][0]), int(p1[0][0][1])
    except:
        return int(p1[0][0]), int(p1[0][1])


# define font and text color
font = cv2.FONT_HERSHEY_SIMPLEX

# define movement threshodls
max_head_movement = 20
movement_threshold = 50
scrolldown_gesture_threshold = 30
scrollup_gesture_threshold = -30

# find the face in the image
face_found = False
frame_num = 0
while not face_found:
    # Take first frame and find corners in it
    frame_num += 1
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, 1.1, 1)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_found = True
    cv2.imshow('Gesture Tracker', frame)
    cv2.waitKey(1)
face_center = x + w / 2, y + h / 3
p0 = np.array([[face_center]], np.float32)


def findFace():
    global p0
    face_found = False
    frame_num = 0
    while not face_found:
        # Take first frame and find corners in it
        frame_num += 1
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 1)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_found = True
        cv2.imshow('Gesture Tracker', frame)
        cv2.waitKey(1)
    face_center = x + w / 2, y + h / 3
    p0 = np.array([[face_center]], np.float32)


# The key combination to check
ctrl_f1 = {keyboard.Key.ctrl_l, keyboard.Key.f1}
ctrl_f2 = {keyboard.Key.ctrl_l, keyboard.Key.f2}
ctrl_alt = {keyboard.Key.ctrl_l, keyboard.Key.alt_l}
ctrl_shft = {keyboard.Key.ctrl_l, keyboard.Key.shift_l}

# The currently active modifiers
current = set()

scrollingEnabled = True

def on_press(key):
    global y_movement, scrollup_gesture_threshold, scrolldown_gesture_threshold, scrollingEnabled
    #recalibrate center of face
    if key == keyboard.Key.esc:
        findFace()
        y_movement = 0
    #set limit for scrolling down
    if key in ctrl_f1:
        current.add(key)
        if all(k in current for k in ctrl_f1):
            print('Setting new limit for scrolling down')
            scrolldown_gesture_threshold = y_movement
    # set limit for scrolling up
    if key in ctrl_f2:
        current.add(key)
        if all(k in current for k in ctrl_f2):
            print('Setting new limit for scrolling up')
            scrollup_gesture_threshold = y_movement
    # reset all scrolling limits (best for recalibrating scroll up or down)
    if key in ctrl_alt:
        current.add(key)
        if all(k in current for k in ctrl_alt):
            print('Removing all Limits for Scrolling')
            scrollup_gesture_threshold = -999
            scrolldown_gesture_threshold = 999
    if key in ctrl_shft:
        current.add(key)
        if all(k in current for k in ctrl_shft):
            scrollingEnabled^=True
            print('Scroll Enabled has been set to ' + str(scrollingEnabled))

#when checking for multiple inputs this removes it from the list if its been released
def on_release(key):
    try:
        current.remove(key)
    except KeyError:
        pass


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()  # start to listen on a separate thread

gesture = False
y_movement = 0
gesture_show_delay_time = 30
gesture_show = gesture_show_delay_time  # number of frames a gesture is shown
calibration_in_progress = False

while True:
    if scrollingEnabled:
        ret, frame = cap.read()
        old_gray = frame_gray.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # cv2.circle(frame, get_coords(p0), 4, (0, 0, 255), -1)
        cv2.circle(frame, get_coords(p0), 4, (255, 0, 0))

        # get the xy coordinates for points p0 and p1
        a, b = get_coords(p0), get_coords(p1)
        y_movement += a[1] - b[1]

        text = 'y_movement: ' + str(y_movement)
        if not gesture: cv2.putText(frame, text, (50, 100), font, 0.8, (255, 0, 0), 2)

        if y_movement > scrolldown_gesture_threshold and gesture_show == gesture_show_delay_time:
            keyboardController.press(Key.page_down)
            keyboardController.release(Key.page_up)
            gesture = 'Down'
        #Checking if scroll up has surpassed threshold, also gives a 60 frame buffer between keypresses.
        if y_movement < scrollup_gesture_threshold and gesture_show == gesture_show_delay_time:
            keyboardController.press(Key.page_up)
            keyboardController.release(Key.page_up)
            gesture = 'Up'
        if gesture and gesture_show > 0:
            cv2.putText(frame, 'Scrolling: ' + gesture, (50, 50), font, 1.2, (255, 0, 0), 3)
            gesture_show -= 1
        if gesture_show == 0:
            gesture = False
            y_movement = 0
            gesture_show = gesture_show_delay_time
        p0 = p1
        cv2.imshow('Gesture Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
