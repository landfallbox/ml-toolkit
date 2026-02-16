"""
工具类使用示例

演示日志管理、检查点管理、配置管理等工具
"""

import os
import tempfile
import torch
from ml_toolkit.utils import Logger, CheckpointManager, ConfigManager

print("工具类使用示例")

# 示例 1: 日志管理器
print("日志管理器")

logger = Logger(name="demo", level="INFO")
logger.info("这是一条信息日志")
logger.warning("这是一条警告日志")
logger.error("这是一条错误日志")

# 示例 2: 配置管理器
print("配置管理器")

config_manager = ConfigManager()
config_manager.set("model.type", "LSTM")
config_manager.set("model.hidden_sizes", [64, 32])
config_manager.set("training.lr", 0.001)
config_manager.set("training.epochs", 100)

print(f"模型类型: {config_manager.get('model.type')}")
print(f"隐藏层大小: {config_manager.get('model.hidden_sizes')}")
print(f"学习率: {config_manager.get('training.lr')}")

# 示例 3: 检查点管理器
print("检查点管理器")

with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint_manager = CheckpointManager(checkpoint_dir=tmpdir)

    # 创建虚拟模型
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters())

    # 保存检查点
    checkpoint_path = checkpoint_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=10,
        metrics={"loss": 0.123, "accuracy": 0.95}
    )

    print(f"检查点已保存到: {checkpoint_path}")

    # 加载检查点
    checkpoint = checkpoint_manager.load_checkpoint(checkpoint_path)
    print(f"加载的检查点信息:")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Metrics: {checkpoint.get('metrics', 'N/A')}")

print("示例完成")




