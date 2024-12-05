from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import pygame
import argparse
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="Path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="", help="Path to alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0, help="Index of webcam on system")
ap.add_argument("-v", "--video", type=str, default="", help="Path to input video file (leave blank for webcam)")
ap.add_argument("-o", "--output", type=str, default="output.avi", help="Path to save output video")
args = vars(ap.parse_args())

# Constants
EYE_AR_THRESH = 0.26
EYE_AR_CONSEC_FRAMES = 48
COUNTER = 0
ALARM_ON = False
START_TIME = None  # 눈을 감기 시작한 시점
WARNING_DURATION = 5  # 눈을 감은 시간이 몇 초 이상일 때 경고

print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

if args["video"]:
    print("[INFO] Loading video file...")
    vs = cv2.VideoCapture(args["video"])
    output_fps = int(vs.get(cv2.CAP_PROP_FPS))
    frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
else:
    print("[INFO] Starting webcam stream...")
    vs = VideoStream(src=args["webcam"]).start()
    time.sleep(1.0)
    output_fps = 20
    frame_width, frame_height = 450, 450

# Initialize video writer for MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
if args["video"]:
    out = cv2.VideoWriter(args["output"], fourcc, output_fps, (frame_width, frame_height))
else:
    out = cv2.VideoWriter(args["output"], fourcc, output_fps, (450, 450))  # Webcam case fixed size

while True:
    if args["video"]:
        success, frame = vs.read()
        if not success:
            break
    else:
        frame = vs.read()

    if frame is None:
        break

    frame = imutils.resize(frame, width=360)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if START_TIME is None:  # 눈을 감기 시작한 시점을 기록
                START_TIME = time.time()

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.daemon = True
                        t.start()

                elapsed_time = time.time() - START_TIME
                if elapsed_time >= WARNING_DURATION:
                    # 눈 감은 시간 표시 (줄을 바꿔서 출력)
                    cv2.putText(frame, f"EYES CLOSED FOR {int(elapsed_time)} SECONDS!", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 졸음 경고 메시지 (다른 위치에 출력)
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False
            START_TIME = None  # 눈을 다시 뜨면 초기화

        # EAR 텍스트를 왼쪽에 표시
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),  # 왼쪽 상단으로 이동
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    out.write(frame)  # Save the frame to the output file
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
if args["video"]:
    vs.release()
else:
    vs.stop()
out.release()
print(f"[INFO] Output saved to {args['output']}")
