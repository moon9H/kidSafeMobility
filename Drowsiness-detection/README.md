# OpenCV를 활용한 졸음운전 탐지

Python, OpenCV, dlib을 사용한 실시간 졸음운전 탐지 시스템입니다.

## 주요 기능
- **EAR(눈 종횡비)**: 눈 감김 상태를 감지하여 졸음 경고.
- **실시간 경고**: 일정 시간 눈을 감고 있으면 경고음을 재생.
- **얼굴 랜드마크 탐지**: dlib의 68포인트 모델 사용.

## 설치 및 실행
1. 필요한 라이브러리 설치:
   ```bash
   pip install opencv-python dlib numpy imutils pygame
   ```
2. 스크립트 실행:
   ```bash
   python detect_drowsiness.py \
       --shape-predictor shape_predictor_68_face_landmarks.dat \
       --alarm alarm.wav
   ```

## 주요 인자
- `--shape-predictor`: 랜드마크 모델 경로.
- `--alarm`: 경고음 파일 경로.

## 결과
- 졸음운전을 실시간으로 감지하며 경고음을 통해 알림.

**안전운전 하세요! 🚗**