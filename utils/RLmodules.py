import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

BUFFER_SIZE = int(1e5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64  # Mini-batch size
GAMMA = 0.99  # Discount factor
TAU = 1e-3  # Target network soft update rate
LR_ACTOR = 1e-4  # Actor learning rate
LR_CRITIC = 1e-3  # Critic learning rate
UPDATE_EVERY = 20  # How often to update the network
EPSILON_START = 1.0  # Initial epsilon for exploration
EPSILON_END = 0.01  # Final epsilon for exploration
EPSILON_DECAY = 0.995  # Epsilon decay rate

class BatteryChargingEnv(gym.Env):
    def __init__(self, N, M, p_max=0.18, tou=None, pos_encoding=None, future_demand=None, max_step=300):
        super(BatteryChargingEnv, self).__init__()
        
        # N: 换电站数量
        # M: 每个换电站电池数量
        # p_max: 最大充电功率
        # tou: 分时电价序列（长度T）
        # l: 预测的未来时隙数量
        
        self.N = N
        self.M = M
        self.p_max = p_max
        self.tou = tou  # 分时电价序列
        self.pos_encoding = pos_encoding
        self.future_demand = future_demand
        
        self.time_step = 0  # 当前时间步
        self.max_step = max_step
        
        self.X = 0  # 每个换电站的充电电池数
        self.Y = future_demand[0]
        self.soc = self.generate_initial_soc(N, M, self.Y[0] + 1)  # 每个电池的充电状态
        self.l = len(future_demand[0])
        self.reward = 0.0
        
        # 动作空间: 每个换电站的充电功率 p_t \in [0, p_max]
        self.action_space = spaces.Box(low=0, high=p_max, shape=(N, M), dtype=np.float32)
        
        # 状态空间: 包括每个电池的充电状态 soc, 未来l个时隙的用户需求 Y_t, 当前时隙的电价 tou_t
        self.state_space = spaces.Dict({
            'soc': spaces.Box(low=0, high=1, shape=(N, M), dtype=np.float32),
            'future demand': spaces.Discrete(200),
            'tou': spaces.Box(low=0, high=2, shape=(self.l,), dtype=np.float32),
        })
    
    def reset(self):
        """环境重置，初始化状态"""
        self.time_step = 0
        self.X = 0  # 每个换电站的充电电池数初始化为0
        self.Y = self.future_demand[0]
        self.soc = self.generate_initial_soc(self.N, self.M, self.Y[0]+2)  # 每个电池的充电状态
        
        # 当前时隙的分时电价
        self.tou_current = self.tou[self.time_step]
        
        tou_state = self.tou.copy()
        tou_state[self.time_step] += self.pos_encoding[self.time_step]
        
        # 返回初始状态
        state = {
            'soc': self.soc,
            'future demand': self.Y,
            'tou': tou_state
        }
        merged_state = np.concatenate([self.soc.flatten(), self.Y.flatten(), tou_state.flatten()])
        return state, merged_state
    
    def generate_initial_soc(self, N, M, K):
        # 创建一个 N x M 的矩阵，里面是 0 到 1 的随机数
        matrix = np.zeros((N, M))
        return matrix
        matrix = np.random.rand(N, M)

        current_ones_count = np.sum(matrix == 1) 
    
        # 如果矩阵中的 1 的数量已经满足 K 个，直接返回
        if current_ones_count == K:
            return matrix
        elif current_ones_count > K:
            require_change = current_ones_count - K
            zero_positions = np.argwhere(matrix == 1)

            # 随机选择 required_ones 个位置，将其替换为 1
            chosen_positions = np.random.choice(len(zero_positions), require_change, replace=False)
            for pos in chosen_positions:
                row, col = zero_positions[pos]
                matrix[row, col] = np.random.uniform(0.3, 0.6)
            return matrix
        
        elif current_ones_count < K:
            # 计算还需要多少个 1
            required_ones = K - current_ones_count

            # 获取当前矩阵中所有的 0 位置
            zero_positions = np.argwhere(matrix < 1)

            # 随机选择 required_ones 个位置，将其替换为 1
            chosen_positions = np.random.choice(len(zero_positions), required_ones, replace=False)
            for pos in chosen_positions:
                row, col = zero_positions[pos]
                matrix[row, col] = 1

            return matrix
    
    def swapping(self):
        N = np.minimum(self.Y[0], self.X)

        ones_positions = np.where(self.soc == 1)

        # 获取1的位置索引
        one_indices = list(zip(ones_positions[0], ones_positions[1]))

        # 随机选择 N 个位置并替换为 [0.3, 0.6] 范围内的随机数
        selected_indices = np.random.choice(len(one_indices), N, replace=False)

        # 替换值
        for idx in selected_indices:
            row, col = one_indices[idx]
            self.soc[row, col] = 0# np.random.uniform(0.3, 0.6)

    def tou_reward(self):
        if self.tou_current == 0.33:
            tou_reward = 4
        elif self.tou_current == 0.76:
            tou_reward = 1
        else:
            tou_reward = -4
        return tou_reward
    
    def step(self, action):
        """根据动作更新环境状态并返回奖励"""
        # 当前时隙的分时电价更新
        # print(action, action <= self.p_max)
        # print()
        assert (action <= self.p_max).all()
        self.tou_current = self.tou[self.time_step % 96]

        # 更新充电状态: soc_{t+1} = soc_t + p_t * 0.25
        total_cost = 0
        total_p = np.sum(action)
        for i in range(self.N):
            for j in range(self.M):
                soc = self.soc[i, j]
                p = action[i, j]
                # 计算电池的新充电状态
                if soc < 0.9:
                    new_soc = soc + p * 0.85 * 0.25
                    total_cost += p * self.tou_current
                    tou_reward = self.tou_reward()
                elif (soc >= 0.9 and soc < 1.0):
                    new_soc = soc + p * 0.85 * 0.25 * 0.6
                    total_cost += p * self.tou_current
                    tou_reward = self.tou_reward()
                else:
                    new_soc = 1
                    tou_reward = 0
                self.soc[i, j] = np.clip(new_soc, 0, 1)
        
        # 更新充电电池数: 当SOC达到1时，增加充电电池数
        self.X = np.sum(self.soc == 1)  # 计算每个换电站的充电电池数量
        average_cost = total_cost / total_p # 平均充电成本
        
        # 计算Q_t = max(X_t - Y_t, 0)，注意这里X_t是每个换电站的充电电池数，Y_t是用户需求
        #Q_t = np.maximum(self.Y[0] - self.X, 0)  # 当前时隙的Q_t = max(X_t - Y_t, 0)
        

        if self.Y[0] - self.X < 0:
            Q_t = (self.Y[0] - self.X) ** 2
        else:
            Q_t = - (self.Y[0] - self.X) ** 2
        
        # 计算奖励 r_t = Q_t - \bar{p}
        self.reward = Q_t - total_cost * 10 + tou_reward
        # self.reward = - total_cost
        
        
        # 增加时间步
        self.time_step += 1
        
        # 判断是否完成任务（例如，最大时隙数或其他条件）
        done = self.time_step >= self.max_step
        
        self.Y = self.future_demand[self.time_step, :]
        
        tou_state = self.tou.copy()
        self.swapping()
        tou_state[self.time_step % 96] += self.pos_encoding[self.time_step % 96]
        
        # 返回新状态
        state = {
            'soc': self.soc,
            'future demand': self.Y,
            'tou': tou_state
        }
        merged_state = np.concatenate([self.soc.flatten(), self.Y.flatten(), tou_state.flatten()])
        #print(merged_state.shape)
        return state, merged_state, self.reward, done, self.Y[0] - self.X, average_cost
    
    def render(self):
        """渲染环境状态"""
        print(f"Time Step: \t{self.time_step}")
        # print(f"SOC: \n{self.soc}")
        print(f"TOU: \t{self.tou_current}")
        print(f"Charging Stations X_t: \t{self.X}")
        print(f"User Demand Y_t: \t{self.Y}")
        print(f"Reward: \t{self.reward}")

class DQN(nn.Module):
    def __init__(self, state_dim, action_space_dim, action_dim):
        super(DQN, self).__init__()
        self.action_space_dim = action_space_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, state_dim*10)
        #self.fc2 = nn.Linear(state_dim*10, state_dim*10)
        self.fc2 = nn.Linear(state_dim*10, action_space_dim*action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        x = self.fc2(x)
        return x.view(-1, self.action_space_dim, self.action_dim)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

class DQNAgent:
    def __init__(self, env, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, lr=0.001, device=None):
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.lr = lr
        self.device = device
        self.loss = nn.MSELoss()

        self.state_dim = int(env.N * env.M + len(env.Y) + len(env.tou))
        self.action_space_dim = env.action_space.shape[0] * env.action_space.shape[1]
        self.action_dim = 10

        self.model = DQN(self.state_dim, self.action_space_dim, self.action_dim).to(device)  # 移动到GPU
        self.target_model = DQN(self.state_dim, self.action_space_dim, self.action_dim).to(device)  # 移动到GPU
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = deque(maxlen=BUFFER_SIZE)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(0, self.env.p_max, self.env.action_space.shape)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # 移动到GPU
        q_values = self.model(state_tensor)
        # q_values = q_values.view(-1, self.action_space_dim, self.action_dim)
        actions = torch.argmax(q_values, dim=2) * 0.02
        return actions.squeeze().detach().cpu().numpy().reshape(self.env.N, self.env.M)  # 转回CPU进行numpy操作

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return torch.tensor(0)

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 将它们转为PyTorch张量并移到GPU
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # 计算Q值
        q_values = self.model(states)
        q_values = q_values.view(-1, self.action_space_dim, self.action_dim)
        q_values = q_values.max(2)[0]

        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            next_q_values = next_q_values.view(-1, self.action_space_dim, self.action_dim)
            next_q_values = next_q_values.max(2)[0]  # 取最大值
            # print((~dones).shape, next_q_values.shape)
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values) * (~dones).unsqueeze(1)

        loss = self.loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def get_action_indices(self, actions):
        indices = []
        for i in range(actions.shape[0]):
            action_cpu = actions[i].cpu().numpy()  # 转换为NumPy数组
            indices.append(np.unravel_index(np.argmax(action_cpu), self.env.action_space.shape))
        return torch.LongTensor(indices).to(self.device)  # 移动到GPU

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


# Define the Actor and Critic Networks
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, state_size-102)
        self.fc2 = nn.Linear(state_size-102, state_size-102)
        self.fc3 = nn.Linear(state_size-102, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Actions should be in [-1, 1]

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat((state, action), dim=1)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)



# Agent class that interacts with the environment
class DDPGAgent:
    def __init__(self, env, state_dim, action_dim, action_limit, batch_size=64, gamma=0.99, tau=0.005, device=None):
        self.env = env
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)
        self.device = device

        # 复制网络参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.action_limit = action_limit

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        action = self.actor(state).cpu().data.numpy().reshape(self.env.N, self.env.M)
        # action = np.clip(action + noise * np.random.randn(len(action)), -self.action_limit, self.action_limit)
        action = np.clip(action + noise * np.random.randn(*action.shape), 0, self.action_limit)

        # print(action)
        return action

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))


    def update_networks(self):
        if len(self.replay_buffer) < self.batch_size:
            return torch.tensor(0.0), torch.tensor(0.0)

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # 确保形状一致
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 计算目标 Q 值
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q_value = rewards + self.gamma * target_q * (1 - dones)

        # 更新 Critic 网络
        # current_q = self.critic(states, actions)
        actions_e = actions.view(self.batch_size, -1)
        current_q = self.critic(states, actions_e)
        #print(current_q.shape, target_q_value.detach().shape)
        critic_loss = nn.MSELoss()(current_q, target_q_value.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新 Actor 网络
        actor_loss = -self.critic(states, self.actor(states)).mean() + 1e-3 * (self.actor(states) ** 2).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)

        return actor_loss.item(), critic_loss.item()

    # def update_networks(self):
    #     if len(self.replay_buffer) < self.batch_size:
    #         return torch.tensor(0.0), torch.tensor(0.0)
    #
    #     batch = random.sample(self.replay_buffer, self.batch_size)
    #     states, actions, rewards, next_states, dones = zip(*batch)
    #     # 将它们转为PyTorch张量并移到GPU
    #     states = torch.FloatTensor(states).to(self.device)
    #     actions = torch.FloatTensor(actions).to(self.device)
    #     rewards = torch.FloatTensor(rewards).to(self.device)
    #     next_states = torch.FloatTensor(next_states).to(self.device)
    #     dones = torch.BoolTensor(dones).to(self.device)
    #
    #     # 计算目标Q值
    #     target_q = self.target_critic(next_states, self.target_actor(next_states))
    #     target_q_value = rewards + (self.gamma * target_q) * ~dones
    #
    #     # 更新Critic网络
    #     actions_e = actions.view(self.batch_size, -1)
    #     current_q = self.critic(states, actions_e)
    #     critic_loss = nn.MSELoss()(current_q, target_q_value)
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     self.critic_optimizer.step()
    #
    #     # 更新Actor网络
    #     actor_loss = -self.critic(states, self.actor(states)).mean()  # 最大化Q值
    #     self.actor_optimizer.zero_grad()
    #     actor_loss.backward()
    #     self.actor_optimizer.step()
    #
    #     # 软更新目标网络
    #     for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
    #         target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    #
    #     for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
    #         target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    #
    #
    #     return actor_loss, critic_loss

class TD3Agent:
    def __init__(self, env, state_dim, action_dim, action_limit, device=None, batch_size = 64, gamma=0.99, tau=0.005, actor_lr=1e-3, critic_lr=1e-3):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_limit = action_limit
        self.policy_delay = 2  # 每 2 次 Critic 更新后才更新 Actor

        # 初始化 Actor 和 Critic
        self.actor = Actor(state_dim, action_dim, action_limit).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_limit).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)

        # 复制权重
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # 经验回放
        # self.replay_buffer = ReplayBuffer(size=1e6)
        self.replay_buffer = deque(maxlen=BUFFER_SIZE)
        self.batch_size = batch_size
        self.update_counter = 0  # 记录 Critic 更新次数

    def select_action(self, state, noise=0.1):
        # state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # action = self.actor(state).cpu().data.numpy().flatten()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().reshape(self.env.N, self.env.M)
        # noise = np.random.normal(0, noise, size=action.shape)
        # action = np.clip(action + noise, 0, self.action_limit)
        add_noise = np.random.uniform(-0.1, 0.1, size=action.shape)
        # 添加噪声进行探索
        action = np.clip(action + add_noise, 0, self.action_limit)

        return action

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update_networks(self, batch_size=128, policy_noise=0.2, noise_clip=0.05):
        if len(self.replay_buffer) < batch_size:
            return None

        # 采样经验
        # states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 目标 Actor 添加噪声
        noise = torch.clamp(torch.randn_like(actions) * policy_noise, -noise_clip, noise_clip).to(self.device)
        # print(noise.shape,self.actor_target(next_states).shape)
        next_actions = self.actor_target(next_states) + noise

        next_actions = torch.clamp(next_actions, 0, self.action_limit)

        # 计算目标 Q 值
        target_q1 = self.critic1_target(next_states, next_actions)
        target_q2 = self.critic2_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)  # 选择较小的 Q 值
        target_q = rewards + (1 - dones) * self.gamma * target_q.detach()

        # 计算当前 Q 值
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        # 计算 Critic 损失
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        # 更新 Critic
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # 延迟更新 Actor
        if self.update_counter % self.policy_delay == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic1_target, self.critic1)
            self.soft_update(self.critic2_target, self.critic2)
        else:
            actor_loss = torch.tensor(0)

        self.update_counter += 1
        return actor_loss.item(), critic1_loss.item(), critic2_loss.item()

    def soft_update(self, target_net, source_net):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


