# AI 기반 통학버스 안전 모니터링 시스템 개발 README
<img width="932" alt="image" src="https://github.com/user-attachments/assets/fafb4776-4cb6-42f3-bed3-c142161b7ba3" />


 ### 아이커넥트, 종합설계프로젝트 1분반 4조 (황문규,박민혁,구성희,이혜원)

 <img width="889" alt="image" src="https://github.com/user-attachments/assets/3909f7ad-4522-41d7-97f4-23b159e622fd" />


## 과제 목표
- 본 과제는 AI 기반의 안전 모니터링 시스템을 도입하여 차량의 내·외부에서 발생할 수 있는 위험 요소를 실시간으로 탐지하고 분석하는 알고리즘을 개발하고자 함. 
- 이를 통해 어린이들의 안전한 통학 환경을 조성하고, 통학버스 운행의 전반적인 안전 수준을 향상시키는 것을 목표로 함.

—

## 주요 기술 및 개발 내용


### 1. 차량 외부 카메라를 활용한 도로 위 위험 요소 탐지

<img width="886" alt="image" src="https://github.com/user-attachments/assets/56f54f8e-a4d0-4925-98b2-926ee6ab767c" />

- **데이터셋 구축 및 라벨링**
  - Custom dataset 총 8000장을 위험요소로 정의한 오토바이, 자전거, 포트홀, 침수구 class로 나누어 라벨링 진행.
  - 데이터 라벨링 도구로는 Roboflow를 사용.

- **모델 개발 및 학습 환경**
  - CNN 기반 YOLOv5 Pytorch 모델을 사용하여 탐지 알고리즘 개발.
  - Colab 환경에서 학습 진행.
  - 학습률, 배치 크기, epoch 등 하이퍼파라미터를 조정하여 모델의 성능을 최적화.

### 2. 운전자의 졸음 운전 및 전방 주시 감지

<img width="880" alt="image" src="https://github.com/user-attachments/assets/06ec782e-d231-41e9-b4cb-0ff12600f83b" />

- **졸음 운전 감지**
  - dlib 라이브러리를 활용하여 눈의 랜드마크 탐지 기능 사용.
  - EAR(Eye Aspect Ratio) 값을 실시간으로 계산하여, 해당 값이 임계값 이하로 1초 이상 유지되면 졸음 상태로 인식하고 알람을 울림.

- **전방 주시 태만 감지**
  - ONNX 라이브러리를 활용하여 yaw와 pitch 각도 계산.
  - SCRFDF를 사용한 FaceDetector로 얼굴 감지 및 얼굴 경계 box 도출.
  - 얼굴 방향이 전방 주시에서 벗어난 상태로 5초 이상 지속될 경우 알람을 울림.

- **개발 환경 및 도구**
  - Pytorch와 Python을 이용하여 전반적인 개발 진행.
  - OpenCV를 활용하여 필터링 및 이미지, 영상 처리 작업 진행.

### 3. 탑승자 Skeleton 감지를 통한 위험 행동 인식
<img width="901" alt="image" src="https://github.com/user-attachments/assets/37c47777-4412-4e21-8684-e55e20951e95" />

- **행동 정의 및 데이터셋 구축**
  - 통학버스 내에서 가능한 행동을 크게 앉기, 걷기, 다투기 세 개의 클래스로 정의.
  - Kaggle을 활용하여 각 클래스별로 5초 분량의 clip 500개 데이터셋 구성.
  - Clip에 대해 프레임별로 키포인트 추출 및 라벨링 작업을 진행하여 학습 데이터 준비.

- **모델 개발 및 학습**
  - Skeleton 기반 행동 인식에 특화된 ST-GCN(Spatial-Temporal Graph Convolutional Network) 모델 활용.
  - 학습을 통해 위험 행동을 실시간으로 탐지 및 분류.

—

## 학습 결과

### 도로 위 위험 요소 탐지 결과

<img width="821" alt="image" src="https://github.com/user-attachments/assets/29cdabbf-9840-4d27-825a-a72369254b70" />


### 운전자 주의 태만 및 졸음 운전 감지 결과
<img width="874" alt="image" src="https://github.com/user-attachments/assets/30af0d11-82df-4dc3-921b-e618d365f774" />


### 탑승자 위험 행동 탐지 결과
<img width="860" alt="image" src="https://github.com/user-attachments/assets/0519bdbb-eaee-4ebd-9bba-1c997399de49" />


—

## 학습 히스토리
- **모델 구성**: 학습 데이터는 도로 위험 요소, 운전자 얼굴 이미지, 탑승자 행동 데이터를 포함하여 총 XX만 건으로 구성.
- **데이터 전처리**: 데이터 확장, 라벨링, 불균형 데이터 보완 수행.
- **모델 학습**: CNN, YOLOv5, ResNet50, OpenPose, ST-GCN 등 다양한 알고리즘 사용.
- **평가 지표**: 정확도, 재현율, F1 스코어 등을 사용하여 성능 평가.

—

## 시스템 통합
- 도로 위험 요소 탐지, 운전자 감지, 탑승자 행동 분석 기능을 통합하여 실시간 모니터링 웹 서비스로 구현.
- 데이터베이스 설계 및 UI/UX 개발 진행 중.

—

## 향후 계획
- 데이터셋 확장 및 알고리즘 최적화를 통한 성능 개선.
- 사용자 피드백 기반 기능 확장 및 상용화 준비.
- 오픈소스 모델 및 데이터 파일 경량화 작업후 임베디드 보드에 탑재.
—

## 참고 문헌 및 데이터셋
- Kaggle Drowsiness Detection Dataset
- 공공 포트홀 데이터셋
- Skeleton 데이터셋
- 내부 구축 데이터셋
