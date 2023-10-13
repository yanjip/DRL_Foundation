import numpy as np
import torch
from utils import torchUtils
import copy
class DQNAgent(object):

    def __init__(self, q_func, optimizer, replay_buffer, batch_size,
                 replay_start_size, n_act, update_target_steps,gamma=0.9, e_greed=0.1,
                 ):
        '''
        :param q_func: Q函数
        :param optimizer: 优化器
        :param replay_buffer: 经验回放器
        :param batch_size: 批次数量
        :param replay_start_size: 开始回放的次数
        :param n_act: 动作数量
        :param gamma: 收益衰减率
        :param e_greed: 探索与利用中的探索概率
        '''
        self.pred_func = q_func
        self.target_func=copy.deepcopy(q_func)
        self.update_target_steps=update_target_steps

        self.global_step = 0

        self.rb = replay_buffer
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size

        self.optimizer = optimizer
        self.criterion = torch.nn.MSELoss()

        self.n_act = n_act  # 动作数量
        self.gamma = gamma  # 收益衰减率
        self.epsilon = e_greed  # 探索与利用中的探索概率

    # 根据经验得到action
    def predict(self, obs):
        obs = torch.FloatTensor(obs)
        Q_list = self.pred_func(obs)
        action = int(torch.argmax(Q_list).detach().numpy())
        return action

    # 根据探索与利用得到action
    def act(self, obs):
        if np.random.uniform(0, 1) < self.epsilon:  # 探索
            action = np.random.choice(self.n_act)
        else:  # 利用
            action = self.predict(obs)
        return action

    def learn_batch(self, batch_obs, batch_action, batch_reward, batch_next_obs, batch_done):

        # predict_Q
        pred_Vs = self.pred_func(batch_obs)
        # pre_Vs: torch.Size([32, 2])
        action_onehot = torchUtils.one_hot(batch_action, self.n_act)
        predict_Q = (pred_Vs * action_onehot).sum(1)
        # predict_Q也是一个tensor【32，1】

        # target_Q
        next_pred_Vs = self.target_func(batch_next_obs)
        best_V = next_pred_Vs.max(dim=1)[0]
        # max返回有两个值，一是value，二是indices
        target_Q = batch_reward + (1 - batch_done) * self.gamma * best_V

        # 更新参数
        self.optimizer.zero_grad()
        loss = self.criterion(predict_Q, target_Q)
        loss.backward()
        self.optimizer.step()

    def learn(self, obs, action, reward, next_obs, done):
        self.global_step += 1
        self.rb.append((obs, action, reward, next_obs, done))

        if len(self.rb)==self.replay_start_size:
            print('--'*20,'经验储备完成******************')

        if len(self.rb) > self.replay_start_size and self.global_step % self.rb.num_steps == 0:
            # 刚开始要先存储一定数量的经验；后面的约束表示每交互rb.num_steps次，才进行一次经验学习
            self.learn_batch(*self.rb.sample(self.batch_size))
            # 这里加星号是因为返回值以元组形式存储，*相当于把元组里的数拆解，再作为参数传入learn_batch

        if self.global_step% self.update_target_steps==0:
            self.sync_target()


    def sync_target(self):
        for target_parma,parma in zip(self.target_func.parameters(),self.pred_func.parameters()):
            target_parma.data.copy_(parma.data)


import pfrl