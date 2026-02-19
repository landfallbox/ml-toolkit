from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_size: int) -> None:
        self.capacity = int(capacity)
        self.state_buf = np.zeros((capacity, state_size), dtype=np.float32)
        self.action_buf = np.zeros(capacity, dtype=np.int64)
        self.reward_buf = np.zeros(capacity, dtype=np.float32)
        self.next_state_buf = np.zeros((capacity, state_size), dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.position
        self.state_buf[idx] = state
        self.action_buf[idx] = action
        self.reward_buf[idx] = reward
        self.next_state_buf[idx] = next_state
        self.done_buf[idx] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            self.state_buf[indices],
            self.action_buf[indices],
            self.reward_buf[indices],
            self.next_state_buf[indices],
            self.done_buf[indices],
        )

