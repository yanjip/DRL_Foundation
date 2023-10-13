# import gym
# env = gym.make('CartPole-v1')  #!!!!!!!!!!!很重要！！！！！！！！！
# for i_episode in range(20):
#     print('*'*40,'游戏[{}/20]开始'.format(i_episode+1),'*'*40)
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print('observation[{}/100]:'.format(t+1) ,end='')
#         print(observation)
#         action = env.action_space.sample()
#         # print(env.step(action))
#         observation, reward, done, info = env.step(action)
#         if done: #done为1表示游戏结束
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

# reset(self)：重置环境的状态，返回观察。
# step(self, action)：推进一个时间步长，返回observation, reward, done, info。
# render(self, mode=‘human’, close=False)：重绘环境的一帧。默认模式一般比较友好，如弹出一个窗口。
# close(self)：关闭环境，并清除内存。

# import numpy as np
# x_data = np.linspace(-2,2,200)[: , np. newaxis]
# print()
#
from gym import envs
for env in envs.registry.all():
    print(env.id)