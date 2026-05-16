# Tron ML Project Plan

## Requirements & Answers

### Q: Where will the trained model be saved?
**A:** Models will be saved in a `checkpoints/` directory with naming convention:
- `tron_dqn_episode_{n}.pth` - checkpoint after n episodes
- `tron_dqn_best.pth` - best performing model
- `tron_dqn_final.pth` - final trained model

### Q: How will the model know opponent location?
**A:** State representation will include opponent information:
- **Grid-based**: Opponent head and trail visible in grid channels
- **Feature-based**: Relative position (dx, dy) to opponent, opponent direction
- **Hybrid**: Both grid spatial info + relative position features

### Q: Can we play against the trained model?
**A:** Yes, will implement:
- `play_vs_ai.py` script for human vs trained model
- Human controls same as 2-player game
- AI uses loaded model to make decisions

### Q: Visual toggle for training vs testing?
**A:** Yes, environment will support:
- `render=False` for headless training (fast)
- `render=True` for testing/demonstration (visual)
- Optional frame skipping for faster training

## Overview
Train machine learning models to play the 2-player Tron game using PyTorch and Deep Reinforcement Learning.

## Analysis of snake-ai-pytorch Reference Implementation

### Key Patterns from snake-ai-pytorch:
1. **Simple State Representation**: 11 features (danger straight/right/left, current direction, food location)
2. **DQN Architecture**: Linear network (input_size=11, hidden_size=256, output_size=3)
3. **Experience Replay**: Deque with max memory of 100,000, batch size of 1,000
4. **Epsilon-Greedy Exploration**: Starts at 80, decreases with games played
5. **Game Interface**: `play_step(action)` returns (reward, game_over, score)
6. **Agent Class**: Handles state extraction, memory management, training
7. **PyTorch Integration**: Model and trainer classes using PyTorch

### Adaptations Needed for Tron:
- **2-Player Competitive**: Not single-player, need opponent modeling or self-play
- **Permanent Trails**: Trails persist until end of round (no expiration)
- **Different Action Space**: Only left/right turns (no straight option)
- **Larger State Space**: Grid-based representation likely needed
- **Multi-Agent Considerations**: Train one agent vs fixed opponent, or self-play

## Implementation Plan

### Phase 1: ML-Friendly Game Environment

#### 1.1 Refactor Game Class
- **Separate rendering from game logic**
- **Add headless mode** (skip pygame display for faster training)
- **Standardize interface**: `reset()`, `step(action)`, `get_state()`
- **Return format**: (observation, reward, done, info)

#### 1.2 Define State Representation
**Option A: Grid-Based (Recommended)**
- 2D array representing the game board
- Channels: [player1_trail, player2_trail, player1_head, player2_head]
- Resolution: Downsample from 800x600 to e.g., 80x60 or 40x30
- Suitable for CNN input
- **Opponent location**: Visible in grid channels (head + trail)

**Option B: Feature-Based (Like snake-ai)**
- Player positions (x, y)
- Player directions (4 one-hot each)
- Trail points (sampled or recent)
- Danger indicators (collision risk in each direction)
- **Opponent relative position**: (dx, dy) distance to opponent
- Opponent direction (4 one-hot)

**Option C: Hybrid (Recommended for best performance)**
- Grid for spatial information (trails, heads)
- Additional features: relative position to opponent, both directions

#### 1.3 Define Action Space
- **Discrete**: [0=turn_left, 1=turn_right]
- **Multi-Discrete** (for both players): [[0,1], [0,1]]

#### 1.4 Define Reward System
- **Win**: +100
- **Loss**: -100
- **Survival**: +0.1 per step (encourage longer games)
- **Proximity to opponent**: Optional small reward for getting closer
- **Trail creation**: Optional small reward for strategic trail placement

#### 1.5 Episode Management
- **Termination conditions**: Player death, wall collision, max steps
- **Reset**: Random starting positions and directions
- **Frame skipping**: Multiple game steps per training step for efficiency

### Phase 2: PyTorch Model Architecture

#### 2.1 CNN Model (for grid-based state)
```python
class TronCNN(nn.Module):
    def __init__(self):
        # Conv layers for spatial features
        # Fully connected for decision making
        # Output: Q-values for actions
```

#### 2.2 MLP Model (for feature-based state)
```python
class TronMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # Similar to snake-ai Linear_QNet
        # Input: feature vector
        # Output: Q-values for actions
```

#### 2.3 Dueling DQN Architecture (Optional)
- Separate value and advantage streams
- Better for state value estimation

### Phase 3: Training Infrastructure

#### 3.1 Experience Replay
- Deque for storing (state, action, reward, next_state, done)
- Batch sampling for training
- Prioritized experience replay (optional)

#### 3.2 DQN Agent
- **Epsilon-greedy exploration**: Decay over training
- **Target network**: Periodic updates for stability
- **Double DQN**: Reduce overestimation bias
- **Training loop**: Collect experience, train on batches

#### 3.3 Training Script
```python
def train():
    agent = TronAgent()
    env = TronEnv(headless=True)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train_short_memory(state, action, reward, next_state, done)
            state = next_state
        
        agent.train_long_memory()
        # Logging, checkpointing, etc.
```

### Phase 4: Multi-Agent Approaches

#### 4.1 Single Agent vs Fixed Opponent
- Train one agent against a simple heuristic opponent
- Easier to implement and debug
- Good starting point

#### 4.2 Self-Play
- Train two agents against each other
- Both improve simultaneously
- More complex but potentially better results

#### 4.3 Population-Based Training
- Maintain population of agents
- Tournament selection
- Evolution of strategies

### Phase 5: Advanced Features

#### 5.1 Curriculum Learning
- Start with shorter trail duration
- Gradually increase difficulty
- Start with slower opponent, then improve

#### 5.2 Imitation Learning
- Collect human gameplay data
- Pre-train model with supervised learning
- Fine-tune with RL

#### 5.3 Model Architecture Improvements
- Attention mechanisms for spatial reasoning
- Recurrent layers for temporal memory
- Graph neural networks for trail structure

#### 5.4 Hyperparameter Optimization
- Learning rate schedules
- Network architecture search
- Exploration strategies

## File Structure

```
tron-ml/
├── tron_game.py          # Original game (keep for reference)
├── tron_env.py           # ML-friendly environment (with render toggle)
├── model.py              # PyTorch models
├── agent.py              # DQN agent
├── train.py              # Training script (headless by default)
├── play_vs_ai.py         # Human vs trained model
├── evaluate.py           # Evaluation/visualization
├── requirements.txt      # Dependencies
├── README.md             # Documentation
└── checkpoints/          # Saved models
    ├── tron_dqn_best.pth
    ├── tron_dqn_final.pth
    └── tron_dqn_episode_*.pth
```

## Dependencies

```
pygame
torch
numpy
gymnasium (optional, for standard RL interface)
matplotlib (for plotting)
tensorboard (optional, for monitoring)
```

## Milestones

1. **Week 1**: ML-friendly environment with headless mode
2. **Week 2**: State representation and reward system
3. **Week 3**: Basic DQN implementation
4. **Week 4**: Training loop and evaluation
5. **Week 5**: Multi-agent / self-play
6. **Week 6**: Advanced features and optimization

## Next Steps

1. Refactor `tron_game.py` to separate game logic from rendering
2. Implement `TronEnv` class with Gymnasium-style interface
3. Choose and implement state representation
4. Implement basic DQN agent
5. Create training script
6. Evaluate and iterate

## References

- snake-ai-pytorch: https://github.com/patrickloeber/snake-ai-pytorch
- Gymnasium: https://gymnasium.farama.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- DQN Paper: https://arxiv.org/abs/1312.5602
