"""
ml-toolkit: Personal Machine Learning Toolkit
可复用的机器学习算法库

这是一个专业的、跨项目的算法库，包含：
- 神经网络模型
- 训练和评估框架
- 性能指标函数
- 数据处理工具
- 通用工具函数

版本：0.1.0
作者：landfallbox

使用示例：
    from ml_toolkit import models, training, evaluation, data_processing, utils

    # 或者直接导入具体的类
    from ml_toolkit.models import LSTM
    from ml_toolkit.training import LSTMTrainer
    from ml_toolkit.evaluation import Evaluator
    from ml_toolkit.data_processing import Normalizer, DatasetLoader
    from ml_toolkit.utils import Logger, CheckpointManager
"""

__version__ = "0.1.0"
__author__ = "landfallbox"

# 导入子模块以便使用 ml_toolkit.models 等方式访问
from . import data_processing, evaluation, models, training, utils

__all__ = [
    "__version__",
    "__author__",
    "models",
    "training",
    "evaluation",
    "data_processing",
    "utils",
]
