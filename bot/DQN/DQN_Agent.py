import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from DQN.DQNet import DQNet
import os
import pickle

random.seed(42)
torch.cuda.manual_seed_all(42)
TAU = 0.005

class DQN_Agent():
    def __init__(self, state_size, gamma, epsilon, lr, batch_size, n_actions,
                 replay_mem_size=50000, min_eps=0.05, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.eps_dec = eps_dec
        self.lr = lr
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.replay_mem = deque(maxlen=replay_mem_size)
        self.model = DQNet(state_size, n_actions)
        self.target_model = DQNet(state_size, n_actions)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

    def update_replay_mem(self, current_state, action, reward, new_state, done):    # Transition = (current_state, action, reward, new_state, done)
        transition = (current_state, action, reward, new_state, done)
        self.replay_mem.append(transition)

    def save_replay_mem(self, path):
        with open('replay_mem.pkl', 'wb') as f:
            pickle.dump(self.replay_mem, f)
    def choose_action(self, current_state):
        if np.random.random() > self.epsilon:
            # Get action from model
            with torch.no_grad():
                q_values = self.model(current_state)
                # print(q_values)
                action = torch.argmax(q_values).item()
                # print(action)
        else:
            # Get random action
            action = np.random.randint(0, self.n_actions)
        return action

    def update_target_model(self):
        # self.target_model.load_state_dict(self.model.state_dict())
        # print("updated")
        target_net_state_dict = self.model.state_dict()
        policy_net_state_dict = self.target_model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_model.load_state_dict(target_net_state_dict)

    def dec_epsilon(self):
        if self.epsilon > self.min_eps:
            self.epsilon -= self.eps_dec

    def train(self):
        if len(self.replay_mem) < self.batch_size:
            return 0
        # Sample a batch from the replay memory of transitions
        batch = random.sample(self.replay_mem, self.batch_size)
        next_state_batch = torch.tensor(np.array([transition[3] for transition in batch]),
                                            device=self.model.device, dtype=torch.float32).to(self.model.device)
        current_state_batch = torch.tensor(np.array([transition[0] for transition in batch]),
                                           device=self.model.device, dtype=torch.float32).to(self.model.device)
        # print(current_state_batch.size())

        action_batch = torch.tensor([transition[1] for transition in batch],
                                    device=self.model.device, dtype=torch.int64).to(self.model.device)
        reward_batch = torch.tensor([transition[2] for transition in batch],
                                    device=self.model.device, dtype=torch.float32).to(self.model.device)

        current_q_values = self.model(current_state_batch).gather(1, action_batch.unsqueeze(1))
        future_q_values = torch.zeros(self.batch_size, device=self.model.device)

        with torch.no_grad():
            future_q_values = self.target_model(next_state_batch).max(1).values

        expected_q_values = reward_batch + future_q_values * self.gamma

        loss = self.criterion(current_q_values, expected_q_values.unsqueeze(1)).to(self.model.device)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.dec_epsilon()
        return loss

    def save_agent(self, current_episode, checkpoint_dir):
        filepath = os.path.join(checkpoint_dir, 'dqn_' + str(current_episode) + '.zip')
        print(filepath)
        torch.save({
            'episode': current_episode,
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'replay_mem': self.replay_mem,
        }, filepath)

    def load_agent(self, filepath):
        state = torch.load(filepath)
        prev_ep = state['episode']
        self.model.load_state_dict(state['model_state_dict'])
        self.target_model.load_state_dict(state['target_model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.replay_mem = state['replay_mem']
        # print(len(self.replay_mem))
        return prev_ep
        
