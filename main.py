from fer import FER
import tensorflow as tf
import cv2
import os
import argparse
from datetime import datetime
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
detector = FER(mtcnn=True)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
COLORS = ((0, 0, 0),
          (100, 0, 0),
          (100, 50, 0),
          (200, 100, 50),
          (255, 150, 50),
          (255, 200, 100),
          (100, 200, 100),
          (50, 100, 200),
          (50, 200, 50))


def color_switch(n: int = 0):
    if n + 1 >= len(COLORS):
        return 0
    else:
        return n + 1


def detect_emotion(img):
    return detector.detect_emotions(img)


def detect_face():
    pass


def detect_smile():
    pass


def draw_rectangle(img, p1: tuple, p2: tuple, color: tuple):
    cv2.rectangle(img, p1, p2, color, 2)


def draw_text(img, text, p, color):
    p = (p[0] + 50, p[1] + 50)
    cv2.putText(img, text, p, cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)


def switch_color():
    pass


def show_image(img, name: str, is_video: bool = False):
    cv2.imshow(name, img)
    if is_video:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return 0
        if cv2.waitKey(1) & 0xFF == ord('s'):
            return 1
    else:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def detect_in_image(img, n):
    emotions = detect_emotion(img)
    for emotion in emotions:
        top_emotion = max(emotion['emotions'], key=emotion['emotions'].get)
        p1 = (emotion['box'][0], emotion['box'][1])
        p2 = (emotion['box'][0] + emotion['box'][2], emotion['box'][1] + emotion['box'][2])
        draw_rectangle(img, p1, p2, COLORS[n])
        draw_text(img, top_emotion, p1, COLORS[n])


def save_image(image):
    cv2.imwrite(datetime.now().strftime("%Y%b%d%H%M%S.jpg"), frame)
    print("Saved selfie as ", datetime.now().strftime("%Y%b%d%H%M%S.jpg"))


parser = argparse.ArgumentParser()
parser.add_argument("--img", "--imgpath", help="path to the image", type=str)
parser.add_argument("--video", "--videopath", help="path to the video file", type=str)
parser.add_argument("--webcam", help="turn webcam caputre mode", action="store_true")
parser.add_argument("--webcam_as", "--webcam_auto_selfie", action="store_true")
args = parser.parse_args()

if args.webcam or args.webcam_as:
    capture = cv2.VideoCapture(0)
    n = 0
    while True:
        check, frame = capture.read()
        frame_copy = copy.deepcopy(frame)
        if not check:
            break
        if args.webcam_as:
            detect_face()
            if detect_smile():
                pass
        else:
            detect_in_image(frame, n)
            ret = show_image(frame, "video", True)
            if ret == 0:
                break
            elif ret == 1:
                save_image(frame)
        n = color_switch(n)

if args.video is not None:
    video = cv2.VideoCapture(args.video)
    while True:
        check, frame = video.read()
        if not check:
            break
        detect_in_image(frame)
        if show_image(frame, "Video", True) == 0:
            break

if args.img is not None:
    img = cv2.imread(args.img)
    detect_in_image(img)
    show_image(img, "Result")

"""
# Read the input image
img = cv2.imread('anger.jpg')
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
emotions = detector.detect_emotions(img)
for emotion in emotions:
    cv2.rectangle(img,(emotion['box'][0],emotion['box'][1]),(emotion['box'][0]+emotion['box'][2],emotion['box'][1]+emotion['box'][2]),(255,120,0),2)
    top_emotion = max(emotion['emotions'], key=emotion['emotions'].get)
    print(top_emotion)
    cv2.putText(img, top_emotion, (emotion['box'][0]+50,emotion['box'][1]+50),cv2.FONT_HERSHEY_COMPLEX,1,(255,200,0),1)
# Display the output
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
"""
