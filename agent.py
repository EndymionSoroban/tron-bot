import torch
import random
import numpy as np
from collections import deque
from model import LinearQNet, ConvQNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9


class TronAgent:
    """DQN Agent for Tron game"""
    def __init__(self, state_type='vector', model_type='linear'):
        """
        Args:
            state_type: 'vector', 'features', or 'grid' - which state representation to use
            model_type: 'linear' for MLP, 'conv' for CNN (use with grid state)
        """
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = GAMMA  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.state_type = state_type
        self.model_type = model_type
        
        # Initialize model based on state type
        if state_type == 'vector':
            # Vector state: 8 features
            self.model = LinearQNet(8, 256, 3)  # 3 actions: straight, left, right
        elif state_type == 'features':
            # Feature state: 22 features (positions, directions, danger, flood fill, etc.)
            self.model = LinearQNet(22, 256, 3)  # 3 actions: straight, left, right
        elif state_type == 'grid':
            # Grid state: 4 channels, 80x60 grid
            if model_type == 'conv':
                self.model = ConvQNet(input_channels=4, grid_size=(80, 60), output_size=3)
            else:
                # Flatten grid for linear model
                self.model = LinearQNet(4 * 80 * 60, 512, 3)
        else:
            raise ValueError(f"Unknown state_type: {state_type}")
        
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, env_state):
        """Extract state from environment state dictionary"""
        if self.state_type == 'vector':
            return env_state['vector']
        elif self.state_type == 'features':
            return env_state['features']
        elif self.state_type == 'grid':
            # Add batch dimension for CNN
            grid = env_state['grid']
            if len(grid.shape) == 3:
                grid = np.expand_dims(grid, axis=0)
            return grid
        else:
            raise ValueError(f"Unknown state_type: {self.state_type}")

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Train on a batch of experiences"""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        
        # Convert to numpy arrays first
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train on a single experience"""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """Get action using epsilon-greedy policy"""
        # No exploration when model is in eval mode (genetic tournaments, play_vs_ai)
        if not self.model.training:
            self.epsilon = 0
        else:
            # Random moves: tradeoff exploration / exploitation
            # Decays from 80 to 5 over ~750 games, providing a long exploration tail
            self.epsilon = max(5, 80 - int(self.n_games * 0.1))
        final_move = [0, 0, 0]  # 3 actions: straight, left, right
        
        if random.randint(0, 200) < self.epsilon:
            # Random action
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Best action from model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def save_model(self, file_name='tron_dqn.pth'):
        """Save the model"""
        self.model.save(file_name)

    def load_model(self, file_name='tron_dqn.pth'):
        """Load the model"""
        return self.model.load(file_name)
