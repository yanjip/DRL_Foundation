import numpy as np

class EposideGreed():
    def __init__(self,n_act,e_greed,decay_rate):
        self.n_act=n_act
        self.epsilon=e_greed
        self.decay_rate=decay_rate

    def act(self,predict_method,obs):
        # 根据探索与利用得到action
        if np.random.uniform(0, 1) < self.epsilon:  # 探索
            action = np.random.choice(self.n_act)
        else:  # 利用
            action = predict_method(obs)
        self.epsilon=max(0.01,self.epsilon-self.decay_rate)
        return action