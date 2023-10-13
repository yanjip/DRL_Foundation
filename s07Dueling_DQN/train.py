'''
@Author  ：Yan JP
@Created on Date：2023/4/12 21:35 
'''
import os
import gym
import numpy as np
import argparse
from s07Dueling_DQN.utils import plot_learning_curve, create_directory
from s07Dueling_DQN.Dueling_DQN import DuelingDQN
# envpath = '/home/xgq/conda/envs/pytorch1.6/lib/python3.6/site-packages/cv2/qt/plugins/platforms'
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=100)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/DuelingDQN/')
parser.add_argument('--reward_path', type=str, default='./output_images/')
parser.add_argument('--epsilon_path', type=str, default='./output_images/')

args = parser.parse_args()


def main():
    env = gym.make('LunarLander-v2')
    # gamma: 折扣因子，用来衡量未来奖励的重要性，一般取0到1之间的值。
    # tau: 目标网络更新的速率，用来平滑目标网络的参数更新，一般取0到1之间的值。
    # epsilon: 探索因子，用来控制智能体在探索和利用之间的平衡，一般取0到1之间的值。
    # eps_end: 探索因子的最小值，用来限制智能体在训练后期的探索行为，一般取0到1之间的值。
    # eps_dec: 探索因子的衰减率，用来控制智能体在训练过程中逐渐减少探索行为，一般取一个很小的正数。
    agent = DuelingDQN(alpha=0.0003, state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
                       fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.99, tau=0.005, epsilon=1.0,
                       eps_end=0.05, eps_dec=5e-4, max_size=1000000, batch_size=256)
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    total_rewards, avg_rewards, eps_history = [], [], []

    for episode in range(args.max_episodes):
        total_reward = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation, isTrain=True)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            total_reward += reward
            observation = observation_

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        eps_history.append(agent.epsilon)
        print('EP:{} reward:{} avg_reward:{} epsilon:{}'.
              format(episode + 1, total_reward, avg_reward, agent.epsilon))

        if (episode + 1) % 50 == 0:
            agent.save_models(episode + 1)

    episodes = [i for i in range(args.max_episodes)]
    plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', args.reward_path)
    plot_learning_curve(episodes, eps_history, 'Epsilon', 'epsilon', args.epsilon_path)


if __name__ == '__main__':
    print("hello")
    main()
