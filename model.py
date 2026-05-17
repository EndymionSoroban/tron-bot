import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class LinearQNet(nn.Module):
    """Simple MLP network for feature-based state representation"""
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearQNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        if os.path.isabs(file_name) or os.path.dirname(file_name):
            dir_name = os.path.dirname(file_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            torch.save(self.state_dict(), file_name)
        else:
            model_folder_path = './checkpoints'
            if not os.path.exists(model_folder_path):
                os.makedirs(model_folder_path)
            file_name = os.path.join(model_folder_path, file_name)
            torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            return True
        model_folder_path = './checkpoints'
        checkpoints_path = os.path.join(model_folder_path, file_name)
        if os.path.exists(checkpoints_path):
            self.load_state_dict(torch.load(checkpoints_path))
            return True
        return False


class ConvQNet(nn.Module):
    """CNN network for grid-based state representation"""
    def __init__(self, input_channels=4, grid_size=(80, 60), output_size=2):
        super(ConvQNet, self).__init__()
        self.grid_height, self.grid_width = grid_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size
        conv_output_size = 128 * self.grid_height * self.grid_width
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        # Conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save(self, file_name='conv_model.pth'):
        if os.path.isabs(file_name) or os.path.dirname(file_name):
            dir_name = os.path.dirname(file_name)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            torch.save(self.state_dict(), file_name)
        else:
            model_folder_path = './checkpoints'
            if not os.path.exists(model_folder_path):
                os.makedirs(model_folder_path)
            file_name = os.path.join(model_folder_path, file_name)
            torch.save(self.state_dict(), file_name)

    def load(self, file_name='conv_model.pth'):
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
            return True
        model_folder_path = './checkpoints'
        checkpoints_path = os.path.join(model_folder_path, file_name)
        if os.path.exists(checkpoints_path):
            self.load_state_dict(torch.load(checkpoints_path))
            return True
        return False


class QTrainer:
    """Trainer class for Q-learning"""
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convert to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Handle batch vs single sample
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: Predicted Q values with current state
        pred = self.model(state)

        # 2: Q_new = r + gamma * max(next_predicted Q value)
        target = pred.clone()
        
        # Calculate max Q value for next state
        q_next = self.model(next_state).max(dim=1)[0]
        
        # Convert done from tuple of bools to tensor
        done_tensor = torch.tensor(done, dtype=torch.float)
        
        # Calculate target Q values
        Q_new = reward + self.gamma * q_next * (1 - done_tensor)
        
        # Get indices for actions taken
        action_indices = torch.argmax(action, dim=1)
        batch_indices = torch.arange(len(action_indices))
        
        # Update target with Q_new for the actions taken
        target[batch_indices, action_indices] = Q_new

        # 3: Calculate loss and optimize
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
