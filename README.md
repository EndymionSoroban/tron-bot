# Tron ML - Deep Reinforcement Learning for Tron Game

Train AI agents to play the classic 2-player Tron game using PyTorch and Deep Q-Learning.

## Features

- **ML-Friendly Environment**: Gymnasium-style interface with headless mode for fast training
- **Multiple State Representations**: Grid-based, feature-based, and vector representations
- **PyTorch Models**: Both CNN (for grid states) and MLP (for feature/vector states)
- **Human vs AI**: Play against your trained models
- **Opponent Awareness**: State includes opponent location and trail information
- **Flexible Training**: Toggle rendering for training vs testing

## Installation

```bash
pip install -r requirements.txt
```

## Files

- `tron_game.py` - Original 2-player game (human vs human)
- `tron_env.py` - ML-friendly environment with render toggle
- `model.py` - PyTorch neural network models
- `agent.py` - DQN agent with experience replay
- `train.py` - Training script
- `play_vs_ai.py` - Human vs trained AI gameplay
- `requirements.txt` - Python dependencies

## Quick Start

### Train a Model

```bash
python3 train.py
```

Training uses headless mode by default for speed. Models are saved to `checkpoints/`:
- `tron_dqn_best.pth` - Best performing model
- `tron_dqn_final.pth` - Final model after training
- `tron_dqn_episode_N.pth` - Checkpoints every 100 episodes

### Play Against Trained AI

```bash
python3 play_vs_ai.py
```

Controls:
- **A** - Turn left
- **D** - Turn right
- **SPACE** - Restart after game over
- **ESC** - Quit

### Play Original Game (Human vs Human)

```bash
python3 tron_game.py
```

Controls:
- **Player 1**: W/D to turn
- **Player 2**: Arrow keys to turn

## Configuration

Edit `train.py` to configure:
- `STATE_TYPE`: 'vector', 'features', or 'grid'
- `MODEL_TYPE`: 'linear' or 'conv'
- `NUM_EPISODES`: Number of training episodes

Edit `play_vs_ai.py` to configure:
- `STATE_TYPE`: Must match trained model
- `MODEL_TYPE`: Must match trained model
- `MODEL_FILE`: Model file to load

## State Representations

### Vector (Default)
- 8 features: positions, directions, relative opponent position
- Fast to compute, suitable for MLP

### Features
- 18 features: positions, directions, opponent info, danger indicators
- More detailed, suitable for MLP

### Grid
- 4 channels × 80×60 grid: player trails and heads
- Spatial information, suitable for CNN

## Reward System

- **Win**: +100
- **Loss**: -100
- **Survival**: +0.1 per step
- **Draw**: -50
- **Timeout**: -10

## Training Tips

1. Start with vector state for faster training
2. Train for at least 500-1000 episodes
3. Monitor win rate - aim for >60%
4. Try different state representations if performance plateaus
5. Use `play_vs_ai.py` to test model behavior

## Project Structure

```
.
├── tron_game.py          # Original game
├── tron_env.py           # ML environment
├── model.py              # Neural networks
├── agent.py              # DQN agent
├── train.py              # Training script
├── play_vs_ai.py         # Human vs AI
├── requirements.txt      # Dependencies
├── README.md             # This file
├── TRON_ML_PLAN.md       # Detailed plan
└── checkpoints/          # Saved models
```

## References

- Based on [snake-ai-pytorch](https://github.com/patrickloeber/snake-ai-pytorch) by Patrick Loeber
- DQN Paper: [Human-level control through deep reinforcement learning](https://arxiv.org/abs/1312.5602)
