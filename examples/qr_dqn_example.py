import gym
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, state, action, next_state, reward, done):
        transition = torch.Tensor([state]), torch.Tensor([action]), torch.Tensor([next_state]), torch.Tensor([reward]), torch.Tensor([done])
        self.memory.append(transition)
        if len(self.memory) > self.capacity: del self.memory[0]

    def sample(self, batch_size):
        sample = random.sample(self.memory, batch_size)
        batch_state, batch_action, batch_next_state, batch_reward, batch_done = zip(*sample)
        
        print(batch_action)
        batch_state = torch.cat(batch_state)
        batch_action = torch.LongTensor(int(batch_action)
        batch_reward = torch.cat(batch_reward)
        batch_done = torch.cat(batch_done)
        batch_next_state = torch.cat(batch_next_state)
        
        return batch_state, batch_action, batch_reward.unsqueeze(1), batch_next_state, batch_done.unsqueeze(1)
    
    def __len__(self):
        return len(self.memory)

class Network(nn.Module):
    def __init__(self, len_state, num_quant, num_actions):
        nn.Module.__init__(self)
        
        self.num_quant = num_quant
        self.num_actions = num_actions
        
        self.layer1 = nn.Linear(len_state, 256)
        self.layer2 = nn.Linear(256, num_actions*num_quant)  

    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        return x.view(-1, self.num_actions, self.num_quant)
    
    def select_action(self, state, eps):
        if not isinstance(state, torch.Tensor): 
            state = torch.Tensor([state])    
        action = torch.randint(0, 2, (1,))
        if random.random() > eps:
            action = self.forward(state).mean(2).max(1)[1]
        return int(action)

eps_start, eps_end, eps_dec = 0.9, 0.1, 500 
eps = lambda steps: eps_end + (eps_start - eps_end) * np.exp(-1. * steps / eps_dec)
env_name = 'MountainCar-v0'
env = gym.make(env_name)

memory = ReplayMemory(10000)
Z = Network(len_state=len(env.reset()), num_quant=2, num_actions=env.action_space.n)
Ztgt = Network(len_state=len(env.reset()), num_quant=2, num_actions=env.action_space.n)
optimizer = optim.Adam(Z.parameters(), 1e-3)

steps_done = 0
running_reward = None
gamma, batch_size = 0.99, 32 
tau = torch.Tensor((2 * np.arange(Z.num_quant) + 1) / (2.0 * Z.num_quant)).view(1, -1)

for episode in range(501): 
    sum_reward = 0
    state = env.reset()
    while True:
        steps_done += 1
        
        action = Z.select_action(torch.Tensor([state]), eps(steps_done))
        next_state, reward, done, _ = env.step(action)

        memory.push(state, action, next_state, reward, float(done))
        sum_reward += reward
        
        if len(memory) < batch_size: break    
        states, actions, rewards, next_states, dones = memory.sample(batch_size)
        
        theta = Z(states)[np.arange(batch_size), actions]
        print(states)
        
        Znext = Ztgt(next_states).detach()
        Znext_max = Znext[np.arange(batch_size), Znext.mean(2).max(1)[1]]
        Ttheta = rewards + gamma * (1 - dones) * Znext_max
        
        diff = Ttheta.t().unsqueeze(-1) - theta 
        loss = huber(diff) * (tau - (diff.detach() < 0).float()).abs()
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        state = next_state
        
        if steps_done % 100 == 0:
            Ztgt.load_state_dict(Z.state_dict())
            
        if done: 
            running_reward = sum_reward  if not running_reward else 0.2 * sum_reward + running_reward*0.8
            break