import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(np.prod(input_dims), 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)

        self.fc5 = nn.Linear(256, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc6 = nn.Linear(256, 256)
        self.bn6 = nn.BatchNorm1d(256)

        self.fc7 = nn.Linear(256, 256)
        self.bn7 = nn.BatchNorm1d(256)

        self.fc8 = nn.Linear(256, 128)
        self.bn8 = nn.BatchNorm1d(128)

        self.fc9 = nn.Linear(128, 64)
        self.bn9 = nn.BatchNorm1d(64)

        self.fc10 = nn.Linear(64, n_actions)

        self.optimizer = optim.AdamW(
            self.parameters(), lr=lr, weight_decay=1e-4)
        self.loss = nn.SmoothL1Loss()
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.view(state.size(0), -1)

        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = F.relu(self.bn8(self.fc8(x)))
        x = F.relu(self.bn9(self.fc9(x)))

        actions = self.fc10(x)
        return actions


class Agent:
    def __init__(self,
                 alpha=0.01,
                 gamma=0.9,
                 epsilon=0.9,
                 lr=0.005,
                 input_dims=[19],
                 batch_size=512,
                 n_actions=5,
                 max_mem_size=1000000,
                 eps_end=0.2,
                 eps_dec=5e-4):

        self.alpha = alpha
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
            self.Q_eval.eval()
            state = torch.tensor([observation], dtype=torch.float32).to(
                self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def train(self, algorithm="bellman"):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.train()
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

        if algorithm == "bellman":
            q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0] * \
                (1 - terminal_batch.float())
        elif algorithm == "td":
            td_error = reward_batch + self.gamma * \
                q_next[batch_index, action_batch] - q_eval
            q_target = q_eval + self.alpha * td_error

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = max(self.eps_end, self.epsilon - self.eps_dec)

    def load_model(self, path):
        self.epsilon = 0
        self.Q_eval.load_state_dict(torch.load(path))
