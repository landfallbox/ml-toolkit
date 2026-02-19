from __future__ import annotations

import random

import numpy as np
import torch
from ml_toolkit.utils import create_optimizer

from src.dqn.network import QNetwork
from src.dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_space: np.ndarray,
        hidden_sizes: list[int],
        learning_rate: float,
        gamma: float,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
        memory_capacity: int,
        batch_size: int,
        target_update_freq: int,
        device: torch.device,
    ) -> None:
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = len(action_space)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        self.policy_net = QNetwork(state_size, self.action_size, hidden_sizes).to(device)
        self.target_net = QNetwork(state_size, self.action_size, hidden_sizes).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = create_optimizer(self.policy_net, "adam", learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(memory_capacity, state_size)
        self.learn_steps = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def get_action_value(self, action_index: int) -> float:
        return float(self.action_space[action_index])

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self) -> float:
        if self.replay_buffer.size < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(loss.item())

    def save_checkpoint(self, path: str, episode: int, best_reward: float | None = None) -> None:
        state = {
            "episode": episode,
            "best_reward": best_reward,
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }
        torch.save(state, path)

