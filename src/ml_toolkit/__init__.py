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
"""

__version__ = "0.1.0"
__author__ = "landfallbox"
__all__ = []

# 延迟导入，避免循环依赖
def __getattr__(name):
    if name == "models":
        from . import models
        return models
    elif name == "training":
        from . import training
        return training
    elif name == "evaluation":
        from . import evaluation
        return evaluation
    elif name == "data_processing":
        from . import data_processing
        return data_processing
    elif name == "utils":
        from . import utils
        return utils
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
