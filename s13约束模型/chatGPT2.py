'''
@Author  ：Yan JP
@Created on Date：2023/4/20 23:02 
'''
import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim


class Critic(nn.Module):
    def __init__(self, input_size, output_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DualPolicyOptimizer():
    def __init__(self, state_size, action_size, lr, l2_reg):
        self.actor = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        self.critic = Critic(state_size, 1)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=l2_reg)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=l2_reg)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")


    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_probs = self.actor(state).to(self.device)
        action = torch.multinomial(action_probs, num_samples=1)
        return action.item()

    def update(self, state_batch, action_batch, reward_batch):
        # Update critic network
        self.critic_optimizer.zero_grad()
        value_batch = self.critic(state_batch).squeeze(-1)
        loss_critic = nn.MSELoss()(value_batch, reward_batch)
        loss_critic.backward()
        self.critic_optimizer.step()

        # Update actor network
        self.actor_optimizer.zero_grad()
        action_probs = self.actor(state_batch)
        log_probs = torch.log(action_probs)
        action_mask = torch.zeros(action_probs.size(), device=self.device)
        action_mask.scatter_(1, action_batch.unsqueeze(1), 1)
        selected_log_probs = (log_probs * action_mask).sum(dim=-1)
        advantage_batch = (reward_batch - value_batch.detach())
        policy_loss = (-selected_log_probs * advantage_batch).mean()
        policy_loss.backward()
        self.actor_optimizer.step()


def main():
    env = gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dual_optimizer = DualPolicyOptimizer(state_size, action_size, lr=1e-3, l2_reg=1e-4)
    replay_buffer = []
    batch_size = 32

    for i_episode in range(1000):
        state = env.reset()
        episode_reward = 0

        while True:
            # Select action and perform it
            action = dual_optimizer.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Store transition

            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

            # Train every batch_size transitions
            if len(replay_buffer) >= batch_size:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(
                    *random.sample(replay_buffer, batch_size))
                state_batch = torch.FloatTensor(state_batch).to(dual_optimizer.device)
                action_batch = torch.LongTensor(action_batch).to(dual_optimizer.device)
                reward_batch = torch.FloatTensor(reward_batch).to(dual_optimizer.device)
                next_state_batch = torch.FloatTensor(next_state_batch).to(dual_optimizer.device)
                done_batch = torch.FloatTensor(done_batch).to(dual_optimizer.device)

                # Compute target value for critic network
                with torch.no_grad():
                    next_value_batch = dual_optimizer.critic(next_state_batch).squeeze(-1)
                    target_value_batch = reward_batch + (1 - done_batch) * 0.99 * next_value_batch

                # Update critic and actor networks
                dual_optimizer.update(state_batch, action_batch, target_value_batch)

                # Clear replay buffer
                replay_buffer = []

            if done:
                print(f"Episode {i_episode + 1} reward: {episode_reward}")
                break
if __name__ == '__main__':
    main()

'''
在主循环中，我们首先将每个转换存储在replay_buffer列表中。然后，当replay_buffer中的转换数量达到batch_size时，
我们从replay_buffer中随机抽样batch_size个转换，并将它们拼接成张量，作为我们的训练数据。
接下来，我们计算critic网络的目标值target_value_batch，并更新critic和actor网络。
最后，我们清空replay_buffer，开始下一个回合的收集转换。

注意：这只是一个简单的示例，用于说明如何在强化学习中使用对偶梯度。
实际应用中，可能需要更复杂的网络和训练策略，以达到更好的性能。
'''