"""
训练框架库
包含通用的训练器基类和训练流程
"""

from .lstm_trainer import LSTMTrainer
from .trainer import Trainer

__all__ = ["Trainer", "LSTMTrainer"]
