import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-4
WEIGHT_DECAY = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_size=128):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.ModuleList([nn.Linear(state_size, hidden_size),
                                      nn.Linear(hidden_size, action_size)])

    def forward(self, state):
        x = state
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        return torch.tanh(self.layers[-1](x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_size=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.ModuleList([nn.Linear(state_size + action_size, hidden_size),
                                      nn.Linear(hidden_size, 1)])

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        return self.layers[-1](x)


'''

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_sizes=(128, 128)):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.ModuleList([nn.Linear(state_size, hidden_sizes[0])])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in zip(hidden_sizes[:-1], hidden_sizes[1:])])
        self.layers.append(nn.Linear(hidden_sizes[-1], action_size))


    def forward(self, state):
        x = state
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        return torch.tanh(self.layers[-1](x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_sizes=(128, 128)):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.ModuleList([nn.Linear(state_size + action_size, hidden_sizes[0])])
        self.layers.extend([nn.Linear(h1, h2) for h1, h2 in zip(hidden_sizes[:-1], hidden_sizes[1:])])
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        return self.layers[-1](x)

class OUNoise:
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.3):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def decrease_sigma(self, delta_sigma):
        self.sigma = max(0.1, self.sigma - delta_sigma)
        print(self.sigma)

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class DDPGAgent():
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.noise = OUNoise(action_size, random_seed)

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        self.actor_lr_scheduler = StepLR(self.actor_optimizer, step_size=1000, gamma=1)
        self.critic_lr_scheduler = StepLR(self.critic_optimizer, step_size=1000, gamma=1)

    def step(self, state, action, reward, next_state, done, episode, decrease_sigma_every_n_episodes, delta_sigma=0.01):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        if done and (episode) % decrease_sigma_every_n_episodes == 0:
            self.noise.decrease_sigma(delta_sigma)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state.reshape(1, -1)).float().to(device)  # Add a batch dimension
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data +
                                    (1.0 - tau) * target_param.data)



