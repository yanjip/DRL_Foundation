'''
@Author  ：Yan JP
@Created on Date：2023/4/13 16:11 
'''
import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from s09ActorCritic.net import *

class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device) # 价值网络
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr) # 价值网络优化器
        self.gamma = gamma

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        # 这里的self.critic(next_states)是用一个固定的目标网络来计算的，它和更新参数的评论家网络是不同的，
        # 所以后面我们需要用detach()方法切断目标网络的梯度，防止它影响到评论家网络的参数。
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones) # 时序差分目标
        td_delta = td_target - self.critic(states) # 时序差分误差

        #它表示对于每个状态states，用演员网络self.actor输出每个动作的"概率"，
        # 然后用actions选择实际执行的动作的概率，再取对数得到log_probs。 这个就是策略梯度
        log_probs = torch.log(self.actor(states).gather(1, actions))

        #td_delta表示每个动作的!!!!!优势函数!!!!!!!!，用detach()方法阻止梯度反向传播。
        # 最后用-log_probs乘以td_delta得到每个动作的损失，再取平均得到演员网络的总损失actor_loss。
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        #需要用评论家网络的输出值来计算演员网络的损失函数，
        # 但是我们不想让演员网络的梯度影响到评论家网络的参数，就需要用detach()方法
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach())) # 均方误差损失函数
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward() # 计算策略网络的梯度
        critic_loss.backward() # 计算价值网络的梯度
        self.actor_optimizer.step() # 更新策略网络参数
        self.critic_optimizer.step() # 更新价值网络参数

if __name__ == '__main__':
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device=torch.device("cpu")

    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
    print("return_list:",return_list)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Actor-Critic on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Actor-Critic on {}'.format(env_name))
    plt.show()
