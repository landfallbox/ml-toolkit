"""
校准分类器使用示例

演示如何使用随机森林校准器
"""

import numpy as np
from sklearn.datasets import make_classification
from ml_toolkit.calibration import RandomForestCalibrator

# 生成示例数据
X, y = make_classification(
    n_samples=200, n_features=10, n_informative=5, n_redundant=2, random_state=42
)

print(f"数据形状: X={X.shape}, y={y.shape}")
print(f"类别分布: {np.bincount(y)}")

# 示例 1: 创建和训练校准分类器
print("随机森林校准分类器")

calibrator = RandomForestCalibrator(n_estimators=100, random_state=42)

# 训练模型
calibrator.fit(X, y)
print("模型训练完成")

# 示例 2: 预测和概率估计
print("预测和概率估计")

# 获取预测结果
predictions = calibrator.predict(X[:10])
print(f"前10个样本的预测: {predictions}")

# 获取预测概率
probabilities = calibrator.predict_proba(X[:10])
print(f"前10个样本的预测概率:\n{probabilities}")

# 示例 3: 特征重要性
print("特征重要性")

feature_importance = calibrator.feature_importances_
print(f"特征重要性: {feature_importance}")
print(f"最重要的特征索引: {np.argsort(feature_importance)[-3:][::-1]}")
