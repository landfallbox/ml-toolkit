"""
异常检测使用示例

演示隔离森林和 LOF 两种异常检测方法
"""

import numpy as np
from ml_toolkit.anomaly_detection import IsolationForestDetector, LOFDetector

# 生成示例数据（包含异常值）
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=(100, 2))
anomalies = np.random.uniform(low=-5, high=5, size=(5, 2))
data = np.vstack([normal_data, anomalies])

print(f"数据形状: {data.shape}")
print(f"正常样本: 100, 异常样本: 5")

# 示例 1: 隔离森林检测
print("隔离森林异常检测")

iso_detector = IsolationForestDetector(contamination=0.05, random_state=42)
iso_predictions = iso_detector.fit_predict(data)

# -1 表示异常, 1 表示正常
n_anomalies = (iso_predictions == -1).sum()
print(f"检测到的异常样本数: {n_anomalies}")
print(f"异常样本索引: {np.where(iso_predictions == -1)[0]}")

# 示例 2: LOF 检测
print("局部异常因子 (LOF) 检测")

lof_detector = LOFDetector(n_neighbors=20, contamination=0.05)
lof_predictions = lof_detector.fit_predict(data)

n_anomalies_lof = (lof_predictions == -1).sum()
print(f"检测到的异常样本数: {n_anomalies_lof}")
print(f"异常样本索引: {np.where(lof_predictions == -1)[0]}")

# 示例 3: 获取异常分数
print("异常分数")

iso_scores = iso_detector.decision_function(data)
print(f"隔离森林异常分数 (前10个): {iso_scores[:10]}")
print(f"分数范围: [{iso_scores.min():.4f}, {iso_scores.max():.4f}]")

