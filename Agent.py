import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(np.prod(input_dims), 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, n_actions)

        self.optimizer = optim.AdamW(
            self.parameters(), lr=lr, weight_decay=1e-4)
        self.loss = nn.SmoothL1Loss()
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.view(state.size(0), -1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)
        return actions


class Agent:
    def __init__(self,
                 gamma=0.9,
                 epsilon=0.9,
                 lr=0.005,
                 input_dims=[14],
                 batch_size=1024,
                 n_actions=4,
                 max_mem_size=1000000,
                 eps_end=0.001,
                 eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(lr, input_dims, n_actions)

        self.state_memory = np.zeros(
            (self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros(
            (self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(
            self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(
            self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation], dtype=torch.float32).to(
                self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def train(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(
            self.state_memory[batch], dtype=torch.float32).to(self.Q_eval.device)
        new_state_batch = torch.tensor(
            self.new_state_memory[batch], dtype=torch.float32).to(self.Q_eval.device)
        reward_batch = torch.tensor(
            self.reward_memory[batch], dtype=torch.float32).to(self.Q_eval.device)
        terminal_batch = torch.tensor(
            self.terminal_memory[batch], dtype=torch.bool).to(self.Q_eval.device)
        action_batch = torch.tensor(
            self.action_memory[batch], dtype=torch.int64).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0] * \
            (1 - terminal_batch.float())

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = max(self.eps_end, self.epsilon - self.eps_dec)
