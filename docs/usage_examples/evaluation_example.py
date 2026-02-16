"""
评估指标使用示例

演示如何计算各种性能指标
"""

import numpy as np
from ml_toolkit.evaluation.metrics import calculate_accuracy

# 生成示例预测和真实标签
np.random.seed(42)
y_true = np.random.randint(0, 2, size=100)
y_pred = np.random.randint(0, 2, size=100)

print(f"真实标签: {y_true[:20]}")
print(f"预测标签: {y_pred[:20]}")

# 计算准确率
print("性能指标")

accuracy = calculate_accuracy(y_true, y_pred)
print(f"准确率 (Accuracy): {accuracy:.4f}")

# 其他指标示例
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1 分数: {f1:.4f}")

# 混淆矩阵
print("混淆矩阵")

cm = confusion_matrix(y_true, y_pred)
print(f"\n{cm}")
print(f"\nTrue Negatives: {cm[0, 0]}")
print(f"False Positives: {cm[0, 1]}")
print(f"False Negatives: {cm[1, 0]}")
print(f"True Positives: {cm[1, 1]}")




