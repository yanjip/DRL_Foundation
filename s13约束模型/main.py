'''
@Author  ：Yan JP
@Created on Date：2023/4/17 16:05
'''
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 定义超参数
gamma = 0.99 # 折扣因子
alpha = 0.01 # 学习率
beta = 0.01 # 对偶学习率
epsilon = 0.1 # 探索系数
lambda_ = 0.1 # 拉格朗日乘子
max_episodes = 1000 # 最大训练回合数
max_steps = 500 # 每回合最大步数

# 创建环境
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0] # 状态维度
action_dim = env.action_space.n # 动作维度

# 创建策略网络
policy_net = nn.Sequential(
    nn.Linear(state_dim, 16),
    nn.ReLU(),
    nn.Linear(16, action_dim),
    nn.Softmax(dim=-1)
)

# 创建值函数网络
value_net = nn.Sequential(
    nn.Linear(state_dim, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# 创建代价函数网络
cost_net = nn.Sequential(
    nn.Linear(state_dim, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

# 定义优化器
policy_optimizer = optim.Adam(policy_net.parameters(), lr=alpha)
value_optimizer = optim.Adam(value_net.parameters(), lr=alpha)
cost_optimizer = optim.Adam(cost_net.parameters(), lr=alpha)

# 定义损失函数
mse_loss = nn.MSELoss()

# 定义约束条件函数，返回是否违反约束的布尔值
def constraint_func(state):
    angle = state[2] # 杆子的倾斜角度
    return abs(angle) > 0.2618 # 约等于15度

# 定义代价函数，返回每次违反约束的代价值
def cost_func(state):
    return constraint_func(state) * 1.0

# 定义策略梯度损失函数，返回策略梯度损失值和对偶梯度损失值
def policy_loss_func(states, actions, rewards, costs):
    # 计算累积折扣奖励和累积折扣代价
    returns = []
    advantages = []
    Gt = 0 #Gt和Ct分别表示当前状态的回报和成本
    Ct = 0
    V=value_net(torch.from_numpy(np.array(states)).float())
    for reward, cost,v1 in zip(reversed(rewards), reversed(costs),V):
        Gt = reward + gamma * Gt
        Ct = cost + gamma * Ct
        returns.append(Gt)
        advantages.append(Gt - v1)
    returns.reverse()
    advantages.reverse()

    # 计算策略梯度损失值和对偶梯度损失值
    policy_loss = 0
    dual_loss = 0
    for state, action, advantage, return_, cost in zip(states, actions, advantages, returns, costs):
        prob = policy_net(torch.from_numpy(np.array(state)).float())[action]
        policy_loss -= torch.log(prob) * advantage - lambda_ * cost # 加入拉格朗日项
        dual_loss += lambda_ * (return_ - cost) - beta * torch.log(torch.tensor(lambda_)) # 使用对数障碍函数

    return policy_loss, dual_loss,returns

# 训练循环
for i in range(max_episodes):
    state = env.reset() # 重置环境，获取初始状态
    states = []  # 存储状态序列
    actions = []  # 存储动作序列
    rewards = []  # 存储奖励序列
    costs = []  # 存储代价序列
    done = False  # 标记是否结束
    for j in range(max_steps):
        if done:
            break
        # 以一定的概率随机探索，否则根据策略网络选择动作
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(policy_net(torch.tensor(np.array([state])).float())).detach().numpy()
            # action = torch.argmax(policy_net(torch.tensor([state]).detach())[0]).numpy()
        # 执行动作，获取下一个状态，奖励和是否结束
        next_state, reward, done, _ = env.step(action)
        # 计算代价值
        cost = cost_func(state)
        # 存储状态，动作，奖励和代价
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        costs.append(cost)
        # 更新状态
        state = next_state

    # 计算总奖励和总代价
    total_reward = sum(rewards)
    total_cost = sum(costs)

    # 打印训练信息
    print(f"Episode {i}, Reward: {total_reward}, Cost: {total_cost}, Lambda: {lambda_}")

    # 计算策略梯度损失值和对偶梯度损失值
    policy_loss, dual_loss,returns = policy_loss_func(states, actions, rewards, costs)

    # 更新策略网络参数
    policy_optimizer.zero_grad()
    policy_loss.backward(torch.ones_like(policy_loss))
    policy_optimizer.step()

    # 更新值函数网络参数
    value_optimizer.zero_grad()
    value_loss = mse_loss(torch.from_numpy(np.array(returns)).float(), value_net(torch.from_numpy(np.array(states)).float()))
    value_loss.backward()
    value_optimizer.step()

    # 更新代价函数网络参数
    cost_optimizer.zero_grad()
    cost_loss = mse_loss(torch.from_numpy(np.array(costs)).float(), cost_net(torch.from_numpy(np.array(states)).float()))
    cost_loss.backward()
    cost_optimizer.step()

    # 更新拉格朗日乘子参数
    lambda_ = max(0, lambda_ + beta * dual_loss.item())

env.close()  # 关闭环境