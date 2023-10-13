'''
@Author  ：Yan JP
@Created on Date：2023/4/13 17:23 
'''
import sys
import os

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

import gym
import numpy as np
import torch
import torch.optim as optim
import datetime
from s10A2C.multiprocessing_env import SubprocVecEnv
from s10A2C.model import ActorCritic
from s10A2C.utils import save_results, make_dir
from s10A2C.utils import plot_rewards

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
algo_name = 'A2C'  # 算法名称
env_name = 'CartPole-v0'  # 环境名称


class A2CConfig:
    def __init__(self) -> None:
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.n_envs = 2  # 异步的环境数目  原本设置的8
        self.gamma = 0.99  # 强化学习中的折扣因子
        self.hidden_dim = 256
        self.lr = 1e-3  # learning rate
        self.max_frames = 30000
        self.n_steps = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlotConfig:
    def __init__(self) -> None:
        self.algo_name = algo_name  # 算法名称
        self.env_name = env_name  # 环境名称
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.save = True  # 是否保存图片


def make_envs(env_name):
    def _thunk():
        env = gym.make(env_name)
        env.seed(2)
        return env

    return _thunk

#测试env！在环境env上测试模型model的表现，重复10次，然后取平均值作为测试奖励。
def ceshi_env(env, model, vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(cfg.device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward


def compute_returns(next_value, rewards, masks, gamma=0.99):
    '''
    逆序遍历是为了计算累积回报，也就是从最后一个奖励开始，依次乘以折扣因子gamma并累加到前面的奖励上，得到每个状态的目标值。
    这样做的原因是为了让目标值反映未来的奖励信息，而不仅仅是当前的奖励。
    逆序遍历可以方便地实现这个计算过程，而且可以节省空间和时间。
    '''
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def train(cfg, envs):
    print('开始训练!')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    env = gym.make(cfg.env_name)  # a single env
    env.seed(10)
    state_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.n
    model = ActorCritic(state_dim, action_dim, cfg.hidden_dim).to(cfg.device)
    optimizer = optim.Adam(model.parameters())
    frame_idx = 0
    test_rewards = []
    test_ma_rewards = [] #这个列表用来记录测试奖励的移动平均值（moving average）。
    state = envs.reset()
    while frame_idx < cfg.max_frames:
        log_probs = [] #这个列表中的每个元素是一个动作的对数概率（log probability）
        values = []
        rewards = []
        masks = [] #每个状态是否结束的标志（masks）
        entropy = 0
        # rollout trajectory
        for _ in range(cfg.n_steps):
            state = torch.FloatTensor(state).to(cfg.device)
            dist, value = model(state)
            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(cfg.device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(cfg.device))
            state = next_state
            frame_idx += 1
            if frame_idx % 100 == 0:
                test_reward = np.mean([ceshi_env(env, model) for _ in range(10)])
                print(f"frame_idx:{frame_idx}, test_reward:{test_reward}")
                test_rewards.append(test_reward)
                if test_ma_rewards:
                    test_ma_rewards.append(0.9 * test_ma_rewards[-1] + 0.1 * test_reward)
                else: #如果test_ma_rewards为空的话，就执行下面的语句。
                    test_ma_rewards.append(test_reward)
                    # plot(frame_idx, test_rewards)
        next_state = torch.FloatTensor(next_state).to(cfg.device)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks)
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach() #防止回报（returns）参与梯度计算
        values = torch.cat(values)
        advantage = returns - values  #计算优势函数=回报-Vpai（critic网络的值）
        actor_loss = -(log_probs * advantage.detach()).mean() #计算演员（actor）的损失函数，它是优势函数加权的动作对数概率的负均值。注意这里要再次断开优势函数与计算图的连接，因为我们只想更新演员网络，而不想更新评论家网络。
        critic_loss = advantage.pow(2).mean()       #计算评论家（critic）的损失函数，它是优势函数的平方均值。这相当于用均方误差（mean squared error）来衡量回报和价值之间的差距。
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy #计算总的损失函数，它是演员损失和评论家损失的加权和，再减去一个熵（entropy）项。熵项是用来增加探索性（exploration）和防止过拟合（overfitting）的，它是动作分布的熵的均值。这里的权重0.5和0.001可以根据不同的任务进行调整。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('完成训练！')
    return test_rewards, test_ma_rewards


if __name__ == "__main__":
    cfg = A2CConfig()
    plot_cfg = PlotConfig()
    envs = [make_envs(cfg.env_name) for i in range(cfg.n_envs)]
    envs = SubprocVecEnv(envs)
    # 训练
    rewards, ma_rewards = train(cfg, envs)
    make_dir(plot_cfg.result_path, plot_cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path)  # 保存结果
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")  # 画出结果