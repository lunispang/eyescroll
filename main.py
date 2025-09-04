import cv2 as cv
import pyautogui
import global_hotkeys

roi_offset = []
reset_offset_flag = True
capture = cv.VideoCapture(0)

enabled = True
height = 640
roi_size = 300 # region of interest size
capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)

show_window = False
move_threshold = 10

if not capture.isOpened():
    print("Video capture not available")
    exit(2)

face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def toggle():
    global enabled
    enabled = not enabled

def reset_offset():
    global reset_offset_flag
    reset_offset_flag = True

def quick_togglereset():
    global enabled, reset_offset_flag
    reset_offset_flag = not enabled
    enabled = not enabled


binds = [
    ["control + shift + 8", quick_togglereset, None, False],
    ["control + shift + 9", toggle, None, False],
    ["control + shift + 0", reset_offset, None, False]
]

global_hotkeys.register_hotkeys(binds)
global_hotkeys.start_checking_hotkeys()

while True:
    ret, frame = capture.read()
    if not ret:
        break

    if roi_offset != [] and not reset_offset_flag:
        min_x = int(roi_offset[0] - roi_size / 2)
        max_x = int(roi_offset[0] + roi_size / 2)
        min_y = int(roi_offset[1] - roi_size / 2)
        max_y = int(roi_offset[1] + roi_size / 2)

        frame = frame[min_y:max_y, min_x:max_x]

    face = face_classifier.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    if len(face) == 1:
        (x, y, w, h) = face[0]
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        midpoint_x = x + w / 2
        midpoint_y = y + h / 2
        local_x = -midpoint_x + frame.shape[1] / 2 # camera is flipped
        local_y = midpoint_y - frame.shape[0] / 2
        if reset_offset_flag:
            if midpoint_x < roi_size / 2 or midpoint_x > frame.shape[1] - roi_size / 2:
                continue
            if midpoint_y < roi_size / 2 or midpoint_y > frame.shape[0] - roi_size / 2:
                continue
            roi_offset = [midpoint_x, midpoint_y]
            reset_offset_flag = False
        if enabled:
            if local_y > move_threshold:
                pyautogui.scroll(-int(local_y - move_threshold), 0, 1)
            if local_y < -move_threshold:
                pyautogui.scroll(-int(local_y + move_threshold), 0, 1)
            # if local_x > move_threshold:
            #     pyautogui.scroll(int(local_x - move_threshold), 1, 0)
            # if local_x < -move_threshold:
            #     pyautogui.scroll(int(local_x + move_threshold), 1, 0)
        if show_window:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
            if enabled:
                cv.rectangle(frame, (0, 0), (8, 8), (255, 0, 255), 4)
            cv.imshow("frame", frame)
            if cv.waitKey(16) == ord("q"):
                break

capture.release()
cv.destroyAllWindows()