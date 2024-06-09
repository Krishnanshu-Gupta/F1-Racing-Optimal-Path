import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RLAgent:
    def __init__(self, state_size, action_size, mutation_rate=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.mutation_rate = mutation_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the Deep Q-Network
        self.policy = DeepQNetwork(state_size, action_size).to(self.device)
        self.target_network = DeepQNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.policy.state_dict())

        # Define the optimizer and loss function
        self.optimizer = optim.Adam(self.policy.parameters())
        self.loss_fn = nn.MSELoss()

        # Replay memory
        self.replay_memory = ReplayMemory(10000)

        # Hyperparameters
        self.batch_size = 64
        self.gamma = 0.99  # Discount factor
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 0.995

    def get_action(self, state, eps=None):
        if eps is None:
            eps = self.eps_start
        if random.random() > eps:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy(state)
            action = q_values.max(1)[1].item()
        else:
            action = random.randrange(self.action_size)
        return action

    def update_policy(self, state, action, reward, next_state, done):
        self.replay_memory.push(state, action, reward, next_state, done)

        if len(self.replay_memory) < self.batch_size:
            return

        batch = self.replay_memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Compute the Q-values for the current state-action pairs
        q_values = self.policy(states).gather(1, actions)

        # Compute the target Q-values for the next state
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute the loss and update the policy network
        loss = self.loss_fn(q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network
        self.update_target_network()

        # Update epsilon for exploration
        self.eps_start = max(self.eps_end, self.eps_start * self.eps_decay)

    def reproduce(self, mutation_rate=None):
        if mutation_rate is None:
            mutation_rate = self.mutation_rate

        new_agent = RLAgent(self.state_size, self.action_size, mutation_rate)
        new_agent.policy.load_state_dict(self.policy.state_dict())

        # Mutate the weights of the new agent
        for param in new_agent.policy.parameters():
            param.data += mutation_rate * torch.randn(param.shape).to(self.device)

        return new_agent

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy.state_dict())

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename))
        self.target_network.load_state_dict(self.policy.state_dict())

class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.memory)