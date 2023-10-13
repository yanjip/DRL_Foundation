import random
import collections
from torch import FloatTensor

class ReplayBuffer(object):
    def __init__(self, max_size, num_steps=1 ):
        self.buffer = collections.deque(maxlen=max_size)
        self.num_steps  = num_steps

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size) #是一个list，元素为元组，每个元组有五个元素
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*mini_batch)
        #比如obs_batch里面存储了若干个obs
        obs_batch = FloatTensor(obs_batch)
        #aciton_batch是batch_size个0,1
        action_batch = FloatTensor(action_batch)
        reward_batch = FloatTensor(reward_batch)
        next_obs_batch = FloatTensor(next_obs_batch)
        done_batch = FloatTensor(done_batch)
        return obs_batch,action_batch,reward_batch,next_obs_batch,done_batch

    def __len__(self):
        return len(self.buffer)

if __name__ == '__main__':
    a=collections.deque(maxlen=3)
    print(a)
    a.append((1,1))
    a.append((2,2))
    a.append((3,3))
    a.append((4,4))
    print(a)
    state, action = zip(*a)
    print(state, action)
