"""
训练框架库
包含通用的训练器基类和训练流程
"""

from .base_trainer import BaseTrainer
from .lstm_trainer import LSTMTrainer

__all__ = ["BaseTrainer", "LSTMTrainer"]
