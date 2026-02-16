# ml-toolkit

一个跨项目的**机器学习算法库**，提供可复用的深度学习和机器学习组件。

## 🚀 快速开始

### 安装

```bash
# 使用 uv 同步依赖（推荐）
uv sync
```

### 运行示例

项目在 `examples/` 目录下提供了各个模块的使用示例：

```bash
# 运行 LSTM 模型示例
python examples/lstm_example.py

# 运行异常检测示例
python examples/anomaly_detection_example.py

# 其他示例同理，详见 `examples/` 目录
```

## 🏗️ 项目结构

```
ml-toolkit/
├── src/ml_toolkit/
│   ├── models/              # 神经网络模型
│   ├── training/            # 训练框架
│   ├── evaluation/          # 评估框架
│   ├── data_processing/     # 数据处理工具
│   ├── anomaly_detection/   # 异常检测
│   ├── calibrated_classifier/  # 校准分类器
│   ├── data_augmentation/   # 数据增强
│   ├── rl/                  # 强化学习
│   └── utils/               # 通用工具
├── examples/                # 使用示例
├── pyproject.toml           # 项目配置
├── uv.lock                  # 依赖锁文件
└── README.md                # 本文件
```

## 💡 核心设计理念

1. **模块化**：每个功能独立成模块，易于维护和扩展
2. **可复用性**：设计为跨项目的通用工具库，支持多种使用场景
3. **灵活配置**：提供参数配置选项，适应不同需求

## 📦 作为依赖导入

### 使用 uv 导入

你可以直接在你的项目中将本仓库作为依赖导入。在你的 `pyproject.toml` 中添加：

```toml
[project]
dependencies = [
    "ml-toolkit @ git+https://github.com/landfallbox/ml-toolkit.git@main",
]
```

然后运行：

```bash
uv sync
```

### 导入和使用

#### 导入整个模块

```python
from ml_toolkit import models, training, evaluation, data_processing, utils
```

#### 导入具体的类

```python
# 模型
from ml_toolkit.models import LSTM

# 训练与评估
from ml_toolkit.training import LSTMTrainer
from ml_toolkit.evaluation import Evaluator

# 数据处理
from ml_toolkit.data_processing import Normalizer, DatasetLoader

# 异常检测
from ml_toolkit.anomaly_detection import IsolationForestDetector, LOFDetector

# 其他模块同理，详见 `src/ml_toolkit/` 各子包的 `__init__.py`
```

## 📄 许可证

MIT License

---

欢迎提交 Issue 和 Pull Request！
