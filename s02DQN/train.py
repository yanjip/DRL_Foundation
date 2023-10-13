import agent,modules
import gym
import torch

class TrainManager(object):
    def __init__(self,env,eposide=500,lr=0.001,gamma=0.1,e_greed=0.9):
        self.env=env
        n_act=env.action_space.n
        n_obs=env.observation_space.shape[0]
        q_func=modules.MLP(n_obs,n_act)
        optimizer=torch.optim.AdamW(q_func.parameters(),lr=lr)
        self.agent=agent.DQNAgent(
            q_func=q_func,
            optimizer=optimizer,
            n_act=n_act,
            gamma=gamma,
            e_greed=e_greed
        )
        self.episode=eposide

    def train_episode(self):
        total_reward=0
        obs=self.env.reset()
        obs=torch.FloatTensor(obs)
        while True:
            action=self.agent.act(obs)
            next_obs,reward,done,info=self.env.step(action)
            next_obs=torch.FloatTensor(next_obs)
            self.agent.learn(obs,action,reward,next_obs,done)
            obs=next_obs
            total_reward+=reward
            if done :break
        return total_reward

    def test_episode(self,):
        total_reward = 0
        obs = self.env.reset()
        obs = torch.FloatTensor(obs)
        while True:
            action = self.agent.act(obs)
            next_obs, reward, done, info = self.env.step(action)
            next_obs = torch.FloatTensor(next_obs)
            obs = next_obs
            total_reward += reward
            self.env.render()
            if done: break
        return total_reward

    def train(self):
        for e in range(self.episode):
            ep_reward=self.train_episode()
            print('Episode %s: reward = %.1f' % (e, ep_reward))

            if e % 100 == 0:
                test_reward= self.test_episode()
                print('test reward = %.1f' % (test_reward))
        pass
if __name__ == '__main__':
    env1 = gym.make("CartPole-v0")
    tm = TrainManager(env1,eposide=1000)
    tm.train()