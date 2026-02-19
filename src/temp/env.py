from __future__ import annotations

import numpy as np
import pandas as pd

from src.dqn.rewards import RewardCalculator


class SequentialDataEnv:
    """
    顺序遍历数据集的类 Gym 环境。

    将 CSV/DataFrame 数据转换为可迭代的 Gym 风格环境，支持 reset/step 接口。
    适用于离线强化学习或批量数据驱动的策略优化。
    """

    def __init__(
        self,
        data: pd.DataFrame,
        state_columns: list[str],
        reward_calculator: RewardCalculator,
    ) -> None:
        self.data = data.reset_index(drop=True)
        self.state_columns = state_columns
        self.reward_calculator = reward_calculator
        self._index = 0

    def reset(self) -> tuple[np.ndarray, dict]:
        self._index = 0
        state = self._get_state(self._index)
        info = {}
        return state, info

    def step(self, action_value: float) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward, info = self.reward_calculator.compute(self._index, action_value)
        next_index = self._index + 1
        terminated = next_index >= len(self.data)

        if terminated:
            next_state = self._get_state(self._index)
        else:
            next_state = self._get_state(next_index)
        self._index = min(next_index, len(self.data) - 1)

        return next_state, reward, terminated, False, info

    def _get_state(self, index: int) -> np.ndarray:
        state = self.data.loc[index, self.state_columns].to_numpy(dtype=np.float32)
        return state

    @property
    def size(self) -> int:
        return len(self.data)

