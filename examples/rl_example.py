"""
强化学习最小示例。

演示如何使用 ml_toolkit.rl 的 SequenceEnv 与 DQNAgent。
"""

import numpy as np
import pandas as pd
import torch

from ml_toolkit.rl import DQNAgent, RewardCalculator, SequenceEnv


class SimpleRewardCalculator(RewardCalculator):
    """简单奖励函数：动作越接近目标值，奖励越高。"""

    def __init__(self, data: pd.DataFrame, target_column: str):
        self.data = data
        self.target_column = target_column

    def compute(self, step_index: int, action_value: float) -> tuple[float, dict]:
        target_value = float(self.data.iloc[step_index][self.target_column])
        error = abs(action_value - target_value)
        reward = -error
        return reward, {"target": target_value, "error": error}


def main() -> None:
    np.random.seed(42)

    data = pd.DataFrame({
        "feature_1": np.random.randn(50),
        "feature_2": np.random.randn(50),
        "target": np.random.uniform(-5, 5, 50),
    })

    state_columns = ["feature_1", "feature_2"]
    reward_calculator = SimpleRewardCalculator(data, "target")
    env = SequenceEnv(data, state_columns, reward_calculator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_space = np.linspace(-5, 5, 11)

    agent = DQNAgent(
        state_size=env.observation_size,
        action_size=len(action_space),
        action_space=action_space,
        device=device,
        learning_rate=0.001,
        gamma=0.99,
        memory_capacity=500,
        batch_size=16,
        target_update=10,
        hidden_size=64,
    )

    state, _ = env.reset()
    total_reward = 0.0

    for _ in range(env.size):
        action_idx = agent.select_action(state, training=True)
        action_value = agent.get_action_value(action_idx)
        next_state, reward, terminated, _, _ = env.step(action_value)

        agent.store_transition(state, action_idx, reward, next_state)
        agent.learn()

        total_reward += reward
        state = next_state

        if terminated:
            break

    print(f"训练完成，总奖励: {total_reward:.4f}")


if __name__ == "__main__":
    main()
