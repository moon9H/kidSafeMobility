import numpy as np

# 주어진 데이터 (클래스별 Precision, Recall, 총 인스턴스 수)
classes = [
    {"name": "class_0", "precision": 0.992, "recall": 0.91, "instances": 89},
    {"name": "class_1", "precision": 0.948, "recall": 0.889, "instances": 305},
    {"name": "class_2", "precision": 0.85, "recall": 0.758, "instances": 60},
    {"name": "class_3", "precision": 0.921, "recall": 0.83, "instances": 271},
]

# TP, FP, FN 계산 및 전체 정확도 계산
total_TP = 0
total_FP = 0
total_FN = 0

for cls in classes:
    recall = cls["recall"]
    precision = cls["precision"]
    instances = cls["instances"]

    # TP와 FN 계산
    TP = recall * instances
    FN = instances - TP

    # FP 계산
    FP = (TP / precision) - TP

    # 총합 업데이트
    total_TP += TP
    total_FP += FP
    total_FN += FN

# TN 계산이 어려운 경우 정확도는 TP, FP, FN 만으로 계산
accuracy = total_TP / (total_TP + total_FP + total_FN)

# 결과 출력
print(f"True Positive (TP): {total_TP:.2f}")
print(f"False Positive (FP): {total_FP:.2f}")
print(f"False Negative (FN): {total_FN:.2f}")
print(f"Accuracy (TP / (TP + FP + FN)): {accuracy:.4f}")