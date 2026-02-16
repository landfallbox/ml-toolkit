"""
数据处理工具使用示例

演示数据归一化、数据集加载等功能
"""

import numpy as np
from ml_toolkit.data_processing import Normalizer, DatasetLoader

# 示例 1: 数据归一化
print("数据归一化示例")

# 创建示例数据
data = np.array([
    [1.0, 100.0],
    [2.0, 200.0],
    [3.0, 300.0],
    [4.0, 400.0]
])

print(f"原始数据:\n{data}")

# 使用 MinMax 归一化
normalizer = Normalizer(method='minmax')
normalized_data = normalizer.fit_transform(data)
print(f"\nMinMax 归一化后的数据:\n{normalized_data}")

# 反归一化
restored_data = normalizer.inverse_transform(normalized_data)
print(f"\n反归一化后的数据:\n{restored_data}")

# 示例 2: 标准化
print("标准化示例")

normalizer_std = Normalizer(method='standard')
standardized_data = normalizer_std.fit_transform(data)
print(f"标准化后的数据:\n{standardized_data}")
print(f"均值: {standardized_data.mean(axis=0)}")
print(f"标准差: {standardized_data.std(axis=0)}")

# 示例 3: 数据集加载器
print("数据集加载器示例")

loader = DatasetLoader()
print("DatasetLoader 已创建，可用于加载 CSV、Parquet 等格式的数据")
# 实际使用: dataset = loader.load_from_file('path/to/data.csv')

