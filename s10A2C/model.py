'''
@Author  ：Yan JP
@Created on Date：2023/4/13 17:23 
'''

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    ''' A2C网络模型，包含一个Actor和Critic
    '''

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        # 用PyTorch的Categorical类来创建一个离散的动作分布，它可以用来采样动作或者计算动作的对数概率。
        dist = Categorical(probs) #它是一种离散的概率分布，可以用来表示一个有限个数的类别的概率。
        return dist, value


class A2C:
    ''' A2C算法
    '''

    def __init__(self, state_dim, action_dim, cfg) -> None:
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.model = ActorCritic(state_dim, action_dim, cfg.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())

    def compute_returns(self, next_value, rewards, masks):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return returns