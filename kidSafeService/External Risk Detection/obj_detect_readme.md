이 README 파일은 Google Colab에서 YOLOv5를 사용하여 객체 탐지를 실행하고, 결과를 처리하여 IoU(교차영역비율)를 사용해 정확도를 계산하는 방법을 설명합니다.

## 사전 요구 사항

1. **Google Colab 환경 설정**: 객체 탐지 작업을 수행하려면 Google Colab에 접근할 수 있어야 합니다.
2. **YOLOv5 및 Ultralytics 라이브러리**: YOLOv5와 필요한 Ultralytics 라이브러리가 설치되어 있어야 합니다.
3. **Google Drive 접근**: 데이터 읽기 및 쓰기를 위해 Google Drive에 접근이 필요합니다.

### 필요한 파이썬 라이브러리

- `os`
- `glob`
- `cv2` (OpenCV)
- `pandas`
- `numpy`
- `matplotlib`

## 환경 설정 방법

1. **Google Drive 마운트**: 데이터셋에 접근하고 결과를 저장하기 위해 Google Drive를 마운트합니다.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Ultralytics 설치**: YOLOv5를 포함하는 Ultralytics 라이브러리를 설치하고 git의 코드를 clone합니다.
    ```sh
    !git clone <repo_url>
    !pip install ultralytics
    ```

3. **이전 결과 제거 (선택 사항)**: 이전 탐지 결과를 제거하여 혼동을 방지합니다.
    ```sh
    !rm -r /content/drive/MyDrive/object_detection/colab/yolov5/runs/danger_detection_v1_0/latest_results
    ```

## YOLOv5 객체 탐지 실행

1. **YOLOv5 디렉토리로 이동**: 모델이 있는 YOLOv5 디렉토리로 이동합니다.
    ```sh
    cd /content/drive/MyDrive/object_detection/colab/yolov5
    ```

2. **YOLOv5 탐지 실행**:
    ```sh
    !python3 detect.py --weights /content/drive/MyDrive/object_detection/weights/best.pt \
        --source /content/drive/MyDrive/object_detection/colab/danger_objects_v1_0/images/ \
        --img 640 --save-txt --project /content/drive/MyDrive/object_detection/colab/yolov5/runs/danger_detection_v1_0 --name latest_results --exist-ok
    ```
   - `--weights`: 사전 학습된 가중치의 경로.
   - `--source`: 입력 이미지 경로.
   - `--img`: 이미지 크기.
   - `--save-txt`: 탐지 라벨을 텍스트 파일로 저장.
   - `--project`: 출력이 저장될 디렉토리.
   - `--name`: 출력 폴더 이름.
   - `--exist-ok`: 기존 디렉토리 허용.

## 정확도 계산

이 섹션에서는 객체 탐지 결과를 처리하여 IoU(교차영역비율)를 기준으로 정확도를 계산합니다.

1. **필요한 라이브러리 임포트**
    ```python
    import os
    import glob
    import numpy as np
    import cv2
    import pandas as pd
    import matplotlib.pyplot as plt
    ```

2. **도움 함수**
    - `box_area(box)`: 바운딩 박스의 면적을 계산합니다.
    - `box_iou_calc(boxes1, boxes2)`: 탐지된 박스와 실제 박스 사이의 IoU를 계산합니다.
    - `process_batch(confusion_matrix, labels, detections, num_classes, iou_threshold=0.5)`: 정확도 평가를 위한 혼동 행렬을 업데이트합니다.

3. **정확도 계산 방법**
    - 파일에서 라벨을 읽고, 예측 결과와 비교하여 TP(참 긍정), FP(거짓 긍정), FN(거짓 부정)을 계산합니다.
    - `process_batch()` 함수는 IoU 임계값(기본값 = 0.5)을 사용하여 실제 라벨과 탐지 결과를 비교합니다.

4. **혼동 행렬 및 정확도 계산 코드   실행**:
    ```python
    original_test_data_label_path   = '/content/drive/MyDrive/object_detection/colab/danger_objects_v1_0/labels/'
    prediction_test_data_label_path = '/content/drive/MyDrive/object_detection/colab/yolov5/runs/danger_detection_v1_0/latest_results/labels/'

    confusion_matrix = np.zeros((5, 5), np.int32)

    original_label_list = glob.glob(os.path.join(original_test_data_label_path, '*.txt'))
    for original_label_filename in original_label_list:
        prediction_label_filename = os.path.join(prediction_test_data_label_path, os.path.basename(original_label_filename))

        # 파일에서 원래 라벨 읽기
        init_labels = []
        with open(original_label_filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.split(' ')
                cls_idx = int(data[0])
                x = float(data[1])
                y = float(data[2])
                w = x+float(data[3])
                h = y+float(data[4])
                init_labels.append([cls_idx, x, y, w, h])

        pred_labels = []
        if os.path.exists(prediction_label_filename):
            with open(prediction_label_filename) as f:
                lines = f.readlines()
                for line in lines:
                    data = line.split(' ')
                    cls_idx = int(data[0])
                    x = float(data[1])
                    y = float(data[2])
                    w = x+float(data[3])
                    h = y+float(data[4])
                    pred_labels.append([cls_idx, x, y, w, h])

            init_labels = np.array(init_labels)
            pred_labels = np.array(pred_labels)
            batch_confusion_matrix = process_batch(confusion_matrix, init_labels, pred_labels, 4, 0.5)
            confusion_matrix = confusion_matrix + batch_confusion_matrix
        else:
            for init_label in init_labels:
                confusion_matrix[4][init_label[0]] += 1

    print(confusion_matrix)

    tp = np.trace(confusion_matrix)
    total = np.sum(confusion_matrix, (0, 1))
    print("Accuracy:", tp/total * 100)
    ```
    - 스크립트는 YOLOv5 객체 탐지 모델의 혼동 행렬과 정확도를 계산하고 출력합니다.

## 결과
- **혼동 행렬**: 4개의 객체 클래스에 대해 TP, FP, FN을 나타내는 `5x5` 혼동 행렬이 계산됩니다.
- **정확도**: 객체 탐지 모델의 최종 정확도가 백분율로 계산됩니다.

## 주의사항
- 입력 이미지, 라벨, 가중치의 파일 경로가 Google Drive 설정과 일치하도록 올바르게 구성되었는지 확인하십시오.
- 특정 탐지 요구 사항에 따라 적절한 IoU 임계값을 설정하십시오.
