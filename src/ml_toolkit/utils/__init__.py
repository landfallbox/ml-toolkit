"""
通用工具库
包含日志、检查点、配置等通用功能
"""

from .checkpoint_manager import CheckpointManager
from .config_manager import ConfigManager
from .logger import Logger
from .loss_factory import create_loss_fn
from .metrics_recorder import MetricsRecorder
from .optimizer_factory import create_optimizer

__all__ = [
    "Logger",
    "CheckpointManager",
    "ConfigManager",
    "MetricsRecorder",
    "create_loss_fn",
    "create_optimizer",
]

# 可选的可视化和超参数优化工具（需要额外依赖）
try:
    from .visualizer import Visualizer
    __all__.append("Visualizer")
except ImportError:
    pass

try:
    from .hyperparameter_optimizer import HyperparameterSpace, BayesianOptimizer
    __all__.extend(["HyperparameterSpace", "BayesianOptimizer"])
except ImportError:
    pass
