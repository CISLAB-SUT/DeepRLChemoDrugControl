#Hoo

#Hoo

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import wrappers
from torch.autograd import Variable
from collections import deque

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

log = False
expl_noise = 0.1

# Step 1: We initialize the Experience Replay memory¶
class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if log:   print('in add mem...', transition)
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    if log: print('in sample mem...')
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind:
      # print('sampled..', self.storage)
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    #if log:   print('batch act1 ', batch_actions)
    batch_actions = np.array(batch_actions)
    if batch_actions.ndim == 1:
        if log:   print('ndim === 1')
        batch_actions = np.expand_dims(batch_actions, axis=1)
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)



# Step 2: We build one neural network for the Actor model and one neural network for the Actor target¶
class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 100)
        self.layer_2 = nn.Linear(100, 30)
        self.layer_3 = nn.Linear(30, action_dim)
        self.max_action = max_action

    def forward(self, x):
        if log: print('in actor forward ...')
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        # x = self.max_action * torch.tanh(self.layer_3(x))
        x = self.max_action * torch.sigmoid(self.layer_3(x)) # [0,5]    not negative dosage
        return x

#Step 3: We build two neural networks for the two Critic models and two neural networks for the two Critic targets¶
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.layer_1 = nn.Linear(state_dim + action_dim, 100)
        self.layer_2 = nn.Linear(100, 30)
        self.layer_3 = nn.Linear(30, 1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(state_dim + action_dim, 100)
        self.layer_5 = nn.Linear(100, 30)
        self.layer_6 = nn.Linear(30, 1)

    def my_forward(self, x, u):
        xu = torch.cat([x, u], 1)
        # Forward-Propagation on the first Critic Neural Network
        y1 = nn.Linear(self.state_dim + self.action_dim, 100)(xu)
        print(y1.shape)
        x1 = F.relu(y1)
        print(x1.shape)
        y1 = nn.Linear(100, 30)(x1)
        x1 = F.relu(y1)
        y1 = nn.Linear(30, 1)(x1)
        # Forward-Propagation on the second Critic Neural Network
        y2 = nn.Linear(self.state_dim + self.action_dim, 100)(xu)
        x2 = F.relu(y2)
        y2 = nn.Linear(100, 30)(x2)
        x2 = F.relu(y2)
        y2 = nn.Linear(30, 1)(x2)
        return y1, y2

    def forward(self, x, u):
        if log:   print('in critic forward ...')
        xu = torch.cat([x, u], 1)
        # Forward-Propagation on the first Critic Neural Network
        y = self.layer_1(xu)
        x1 = F.relu(y)
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1


# Steps 4 to 15: Training Process
# Building the whole Training Process into a class

class TD3(object):

    def __init__(self, state_dim, action_dim, max_action, device):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action
        self.action_dim = action_dim
        self.device = device

    def select_action(self, state, epsilon):
        if log:   print('in sel...', state)
        #print('in sel act... eps ', epsilon)

        if random.random() < epsilon:
            action = np.random.random()*self.max_action
            if expl_noise != 0:
                action = (action + np.random.normal(0, expl_noise, size= self.action_dim)).clip(0, self.max_action)[0]
            if log:   print('explore ', action)
        else:
            state = torch.Tensor(state.reshape(1, -1)).to(self.device)
            action = (self.actor(state).cpu().data.numpy().flatten())[0]

            if expl_noise != 0:
                action = (action + np.random.normal(0, expl_noise, size=self.action_dim)).clip(0, self.max_action)[0]
            if log:   print('exploit ', action)
        return action

    def train(self, replay_buffer, iterations, batch_size, discount=0.99, tau=0.005, policy_noise=0.2,
              noise_clip=0.5, policy_freq=2):
        if log: print('in train...')
        for it in range(iterations):

            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            # print('in td3 train...', batch_size, batch_actions.shape, batch_states.shape)
            state = torch.Tensor(batch_states).to(self.device).type(dtype)
            next_state = torch.Tensor(batch_next_states).to(self.device).type(dtype)
            #if log:   print('batch act ', batch_actions)
            # action = torch.Tensor(batch_actions).to(self.device)
            action = (torch.from_numpy(np.asarray(batch_actions))).to(self.device).type(dtype)
            reward = torch.Tensor(batch_rewards).to(self.device).type(dtype)
            done = torch.Tensor(batch_dones).to(self.device).type(dtype)

            # Step 5: From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(next_state)

            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(0, self.max_action)
            if log:   print('actor target ', next_action)

            # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)

            # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(state, action)

            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % policy_freq == 0:
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                # my actor_loss = F.mse_loss(self.critic.Q1(state, self.actor(state)))
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Making a save method to save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    # Making a load method to load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
