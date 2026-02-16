"""
数据增强使用示例

演示序列数据的噪声增强和重采样增强
"""

import numpy as np
from ml_toolkit.data_augmentation import (
    SequenceNoiseAugmentor,
    SequenceResamplingAugmentor,
)

# 生成示例时间序列数据
np.random.seed(42)
sequence = np.random.randn(100, 5)  # 100 时间步，5 个特征
labels = np.random.randint(0, 2, size=sequence.shape[0])
time_indices = np.arange(sequence.shape[0])

print(f"原始序列形状: {sequence.shape}")
print(f"原始序列前5行:\n{sequence[:5]}")

# 示例 1: 序列噪声增强
print("序列噪声增强")

noise_augmenter = SequenceNoiseAugmentor(noise_scale=0.1)
augmented_noise, augmented_labels, augmented_time = noise_augmenter.augment(
    sequence, labels, time_indices
)

print(f"增强后序列形状: {augmented_noise.shape}")
print(f"增强后序列前5行:\n{augmented_noise[:5]}")
print(
    f"与原始序列的差异 (RMSE): {np.sqrt(np.mean((sequence - augmented_noise[: sequence.shape[0]]) ** 2)):.4f}"
)

# 示例 2: 序列重采样增强
print("序列重采样增强")

resample_augmenter = SequenceResamplingAugmentor(target_ratio=0.8)
augmented_resample, resampled_labels, resampled_time = resample_augmenter.augment(
    sequence, labels, time_indices
)

print(f"增强后序列形状: {augmented_resample.shape}")
print(f"原始长度: {sequence.shape[0]}, 重采样长度: {augmented_resample.shape[0]}")
print(f"增强后序列前5行:\n{augmented_resample[:5]}")

# 示例 3: 多次增强
print("多次增强组合")

augmented_combined = augmented_noise.copy()
combined_labels = augmented_labels.copy()
combined_time = augmented_time.copy()
for i in range(3):
    augmented_combined, combined_labels, combined_time = noise_augmenter.augment(
        augmented_combined, combined_labels, combined_time
    )

print(f"多次增强后的序列形状: {augmented_combined.shape}")
print(
    f"多次增强后与原始序列的差异 (RMSE): {np.sqrt(np.mean((sequence - augmented_combined[: sequence.shape[0]]) ** 2)):.4f}"
)
