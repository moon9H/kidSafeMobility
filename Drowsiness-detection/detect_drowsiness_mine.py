from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from collections import deque
import numpy as np
import pygame
import argparse
import imutils
import time
import dlib
import cv2


def sound_alarm(path):
    """알람 소리를 재생하는 함수"""
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()


def eye_aspect_ratio(eye):
    """눈의 종횡비(EAR)를 계산하는 함수"""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# 명령줄 인자 정의
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="Path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="", help="Path to alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0, help="Index of webcam on system")

args = vars(ap.parse_args())

EYE_AR_THRESH = 0.26
EYE_AR_CONSEC_FRAMES = 48
COUNTER = 0
ALARM_ON = False

# EAR 히스토리 및 눈 위치 히스토리
EAR_HISTORY = deque(maxlen=5)
LEFT_EYE_HISTORY = deque(maxlen=5)
RIGHT_EYE_HISTORY = deque(maxlen=5)

# 얼굴 탐지기 및 랜드마크 예측기 로드
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



# 프레임 반복 처리
while True:
    if args["video"]:
        ret, frame = cap.read()
        if not ret:
            break
    else:
        frame = cap.read()

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) > 0:  # 얼굴 탐지 성공
        rect = rects[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # EAR 계산
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        EAR_HISTORY.append(ear)
        smoothed_ear = np.mean(EAR_HISTORY)

        # 눈 위치 업데이트
        LEFT_EYE_HISTORY.append(leftEye)
        RIGHT_EYE_HISTORY.append(rightEye)

        # 눈 윤곽 그리기
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 졸음 탐지
        if smoothed_ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.daemon = True
                        t.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, "EAR: {:.2f}".format(smoothed_ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # 프레임 표시
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
if args["video"]:
    cap.release()
if args["output"]:
    out.release()
cv2.destroyAllWindows()
