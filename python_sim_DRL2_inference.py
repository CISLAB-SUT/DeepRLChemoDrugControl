# In the name of God


import matplotlib.pyplot as plt
import random
import numpy as np
from TD3 import TD3, ReplayBuffer
import torch
import os
import math
from scipy.integrate import odeint



log = False
attempt=3
seed = 0
max_mem_size = 1e6
eval_freq = 20  # 5e3 # How often the evaluation step is performed (after how many timesteps)
evaluations = []
save_models = True  # Boolean checker whether or not to save the pre-trained model
# expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
batch_size = 500#500  # Size of the batch
discount = 0.7  # gamma... Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005  # Target network update rate
policy_noise = 0.2  # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5  # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2  # Number of iterations to wait before the policy network (Actor model) is updated
action_dim = 1
state_dim = 4  # ? or 4 => change experineces in replay mem
max_action = 5 # max value of actions. max medicine dosage

episodes = 30  # 1000
max_steps = 2000  # 6001#5e5 . max steps in each episode, after which we terminate the episode if not reached to goal
max_train_steps = 50;
warm_up_episodes = 4  # The early episodes. in warmup, don't decay epsilon. so that we select actions randomly.#######odeint
a1=0.2;
a2=0.3;
a3=0.1;
b1=1;
b2=1;
c1=1;
c2=0.5;
c3=1;
c4=1;
d1=0.2;
d2=1;
r1=1.5;
r2=1;
s=0.33;
alfa=0.3;
ro=0.01;
folder_path1 = './results_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s' % (str(attempt),str(seed), str(batch_size),str(discount),str(tau),str(policy_freq),str(max_action),str(episodes),str(max_steps),str(max_train_steps),str(warm_up_episodes))
folder_path2 = './pytorch_models_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s' % (str(attempt),str(seed), str(batch_size),str(discount),str(tau),str(policy_freq),str(max_action),str(episodes),str(max_steps),str(max_train_steps),str(warm_up_episodes))

env_name = 'DRL_tumor'
file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action, device=device)
replay_buffer = ReplayBuffer(max_size=max_mem_size)


class SimulinkPlant:

    def __init__(self, modelName='Learning'):
        np.random.seed(seed)
        random.seed(seed)
        self.modelName = modelName  # The name of the Simulink Model (To be placed in the same directory as the Python Code)

        self.goalstate = 1e-4  # x2. == N um of tomur cells be zero.
        self.inference_zeroActionState=0.1
        self.t_s = 1.0
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.95
        self.episodes = 20
        self.max_steps = 2000#6001#5e5 . max steps in each episode, after which we terminate the episode if not reached to goal
        self.warm_up_episodes = 8  # The early episodes. in warmup, don't decay epsilon. so that we select actions randomly.

        #self.states = [0, 0.0063, 0.0125, 0.025, 0.035, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        #           0.55, 0.6, 0.65, 0.7, 0.8, 0.9]

    def reset(self):

         return np.array([0.6, 0.5, 1, 0])  # state1, state, state3, state4
         #return np.array([0.57578, 0.34141, 0.61721, 1.08178])  # state1, state, state3, state4

    def rewardFunc2(self, state, nstate):
        if nstate <= self.goalstate:
            return 0
        if nstate < state:
            result = -np.log2(1 - (state - nstate) / state)
        else:
            result = 0
        return result

    def pend(self, obs, t, action):
        x1, x2, x3, x4 = obs
        u = action
        dx1 = r2 * x1 * (1 - b2 * x1) - c4 * x1 * x2 - a3 * x1 * x4;
        dx2 = r1 * x2 * (1 - b1 * x2) - c2 * x3 * x2 - c3 * x2 * x1 - a2 * x2 * x4;
        dx3 = s + ro * x3 * x2 / (alfa + x2) - c1 * x3 * x2 - d1 * x3 - a1 * x3 * x4;
        dx4 = -d2 * x4 + u;
        dydt = [dx1, dx2, dx3, dx4]
        return dydt

        #######odeint
    def step(self, obs, action=0):
        if log:   print('in step...')
        t = np.linspace(0, self.t_s, 11)
        sol = odeint(self.pend, obs, t, args=(action,))
        sol = np.transpose(sol)
        x1 = sol[0]
        x2 = sol[1]
        x3 = sol[2]
        x4 = sol[3]
        nstate = x2[-1]
        nstate1 = x1[-1]
        nstate3 = x3[-1]
        nstate4 = x4[-1]
        done = 0
        if nstate <= self.goalstate:
            done = 1
        reward = self.rewardFunc2(obs[1], nstate)  # it can be replaced with the reward function
    #    return np.array([nstate1, nstate, nstate3, nstate4]), reward, done
        return np.array([nstate1, nstate, nstate3, nstate4]), np.array([x1, x2, x3, x4]), reward, done


    # We make a function that evaluates the policy by calculating its average reward over 10 episodes
    def evaluate_policy(self, policy, eval_episodes=4):
        if log:   print('in eval...')
        avg_reward = 0.
        for _ in range(eval_episodes):
            obs = self.reset()#np.array([0.6, 0.5, 1, 0])
            done = 0
            steps = 0
            while done == 0 and steps<400:
                steps+=1
                if log:   print('in eval...', obs)
                action = policy.select_action(obs, epsilon=0)
                nextStates, reward, done = self.step(obs, action)
                obs = nextStates
                avg_reward += reward
        avg_reward /= eval_episodes
        print("---------------------------------------")
        print("Average Reward over the Evaluation Step: %f" % (avg_reward))
        return avg_reward


    def inference(self):
        print('in inference...')
        obs = self.reset()
        #obs = np.array([0.58808, 0.38989, 0.66909, 0.04703])
        steps = 0
        x11, x22, x33, x44 = [], [], [], []
        done = 0
        dosage = []

        # x11.append(obs[0])
        # x22.append(obs[1])
        # x33.append(obs[2])
        # x44.append(obs[3])
        # dosage.append(0)

        while done == 0 and steps<self.max_steps:
            if obs[1]>self.inference_zeroActionState:
                action = policy.select_action(obs, epsilon=0)
            else:
                action=0
            dosage.append(action)

            obs,state_values, reward, done = self.step(obs, action)
            print('steps {}, state {}, action {}, reward {} , done {}'.format(steps, obs, action, reward, done))

            # x11.append(obs[0])
            # x22.append(obs[1])
            # x33.append(obs[2])
            # x44.append(obs[3])
            #x11.append(state_values[0])
            for abc in state_values[0]:
                x11.append(abc)
            #x11+=state_values[0].tolist()
            for abc in state_values[1]:
                x22.append(abc)
            for abc in state_values[2]:
                x33.append(abc)
            for abc in state_values[3]:
                x44.append(abc)

            steps = steps + 1
        dat = np.array([x11, x22, x33, x44])
        dat = dat.T
        np.savetxt('./'+folder_path1+'/graphs_states.txt', dat, delimiter=',',fmt='%1.5f')
        dat = np.array([dosage])
        dat = dat.T
        np.savetxt('./'+folder_path1+'/graphs_dosage.txt', dat, delimiter=',', fmt='%1.5f')

        fig, axs = plt.subplots(5)

        axs[0].plot(x11, 'b', label='x1')
        axs[1].plot(x22, 'b', label='x2')
        axs[2].plot(x33, 'b', label='x3')
        axs[3].plot(x44, 'b', label='x4')
        axs[4].plot(dosage, 'b', label='u')
        #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Topics", fontsize='large', labelspacing=0.6,
        #           fancybox=True)
        #plt.title('num of different cells in time')
        plt.show()
        #plt.plot(dosage)
        #plt.title('dosage')
        #plt.show()


Simul = SimulinkPlant(modelName="Learning")
# Establishes a Connection

policy.load(file_name, folder_path2)
replay_buffer.storage = np.load(folder_path1+"/TD3_DRL_tumor_"+str(seed)+".npy", allow_pickle=True)
print('mem fill len: ', replay_buffer.storage)

#Simul.simulate()

Simul.inference()
