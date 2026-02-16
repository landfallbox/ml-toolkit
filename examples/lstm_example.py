"""
LSTM 模型使用示例

演示如何创建、初始化和使用 LSTM 模型
"""

import torch
import torch.nn as nn
from ml_toolkit.models import LSTM

# 创建 LSTM 模型
model = LSTM(input_size=10, hidden_sizes=[64, 32], output_size=2, batch_first=True, dropout=0.2)

print(f"Model: {model}")

# 创建虚拟输入数据
# 形状: (batch_size, seq_length, input_size)
x = torch.randn(32, 20, 10)

# 前向传播
output = model(x)
print(f"Output shape: {output.shape}")  # 应为 (32, 2)

# 模型参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
