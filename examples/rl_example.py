"""
强化学习 Agent 使用示例

演示如何使用 DQN Agent 和 ReplayBuffer
"""

import numpy as np
import torch
from ml_toolkit.rl.agents import DQNAgent, ReplayBuffer

# 创建超参数
state_size = 4
action_size = 2
memory_size = 1000
batch_size = 32
gamma = 0.99
lr = 0.001

print(f"状态空间大小: {state_size}")
print(f"动作空间大小: {action_size}")

# 示例 1: 创建经验回放缓冲区
print("经验回放缓冲区")

replay_buffer = ReplayBuffer(memory_size)
print(f"创建大小为 {memory_size} 的回放缓冲区")

# 添加一些经验
for _ in range(100):
    state = np.random.randn(state_size)
    action = np.random.randint(0, action_size)
    reward = np.random.randn()
    next_state = np.random.randn(state_size)
    done = np.random.rand() > 0.9

    replay_buffer.add(state, action, reward, next_state, done)

print(f"添加了 100 条经验，缓冲区大小: {len(replay_buffer)}")

# 采样批次
print("采样批次")

if len(replay_buffer) >= batch_size:
    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = batch
    print(f"采样批次大小: {batch_size}")
    print(f"States 形状: {states.shape}")
    print(f"Actions 形状: {actions.shape}")
    print(f"Rewards 形状: {rewards.shape}")
    print(f"Next states 形状: {next_states.shape}")
    print(f"Dones 形状: {dones.shape}")

# 示例 2: 创建 DQN Agent
print("DQN Agent")

device = "cuda" if torch.cuda.is_available() else "cpu"
agent = DQNAgent(state_size=state_size, action_size=action_size, device=device, lr=lr)

print(f"DQN Agent 已创建 (device: {device})")
print(f"网络: {agent.q_network}")

# 示例 3: Agent 动作选择
print("Agent 动作选择")

state = np.random.randn(state_size)
epsilon = 0.1  # 探索率

action = agent.select_action(state, epsilon)
print(f"选择的动作: {action}")

# 多次选择动作
actions = [agent.select_action(np.random.randn(state_size), epsilon) for _ in range(10)]
print(f"10 个随机状态的动作选择: {actions}")
