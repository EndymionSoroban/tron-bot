# Tron Bot ML Framework

## Overview
This repository contains a Machine Learning framework for training AI agents to play the classic game of Tron (Light Cycles). The framework provides a PyGame-based environment, Deep Q-Network (DQN) agents, and multiple training methodologies including standard reinforcement learning, self-play, and genetic algorithms.

## How It Works

### 1. Environment (`tron_env.py`)
Provides a custom reinforcement learning environment similar to OpenAI Gym.
- **State Representations:** 
  - `vector`: Simple vector containing player positions and directions.
  - `features`: Snake-AI style features (distances to walls/trails in 3 directions, current facing direction, normalized positions). This is the default.
  - `grid`: A 2D grid representation intended for Convolutional Neural Networks.
- **Mechanics:** Players move continuously. Colliding with walls, your own trail, or the opponent's trail results in a loss (-100 reward). Surviving gives a small positive reward (+0.1), and winning gives +100.

### 2. Neural Networks (`model.py`)
- **LinearQNet:** A Multi-Layer Perceptron (MLP) used for `vector` and `features` state representations.
- **ConvQNet:** A Convolutional Neural Network used for `grid` state representations.
- **QTrainer:** Implements the Q-learning optimization step using Mean Squared Error (MSE) loss and the Adam optimizer.

### 3. Agent (`agent.py`)
- Implements a DQN Agent with replay memory (experience replay).
- Uses an epsilon-greedy policy for action selection (exploration vs. exploitation).
- Contains logic for both short-term memory training (single step) and long-term memory training (batch of experiences).

### 4. Training Methods
- **Standard RL (`train.py`):** Trains a single AI agent against a simple heuristic-based opponent (which tries to avoid walls randomly).
- **Self-Play (`train_self_play.py`):** Trains two agents simultaneously by having them play against each other in a zero-sum setting.
- **Genetic Algorithm (`train_genetic.py`):** Uses an evolutionary approach. A population of models plays tournament-style matches. Top performers are kept (elitism), while the rest are generated via crossover and mutation of weights.

### 5. Playback (`play_vs_ai.py`)
- Allows a human player (using A/D keys) to play against a trained AI model.

## Issues Found

During the review, a few issues were identified, particularly around how runs and checkpoints are saved:

1. **Genetic Algorithm Saves to Root Directory:**
   In `train_genetic.py`, output directories are defined at the top of the file:
   ```python
   GENETIC_RUNS_DIR = 'genetic_runs'
   GENETIC_WINNER_DIR = 'genetic_winner'
   ```
   However, they are completely ignored in the code. Models are saved directly to the root directory using `torch.save(..., f'genetic_best_gen_{generation}.pth')` instead of being placed in the designated folders.

2. **Progress Plots Clutter Root Directory:**
   Both `train.py` and `train_self_play.py` save their progress plots (`training_progress.png` and `self_play_progress.png`) directly to the root directory. Over time, multiple runs will overwrite each other, and it clutters the root folder. It would be better to save these into timestamped `runs/` folders.

3. **Inconsistent Model Saving:**
   While `agent.py` uses `model.save()` (which correctly places models inside a `./checkpoints` directory), `train_genetic.py` bypasses this mechanism entirely and uses `torch.save()` directly.

4. **Missing Checkpoint Loading Support in Genetic Training:**
   `train_genetic.py` lacks a mechanism to load a previous population or checkpoint. If a long training run is interrupted, it must start over from generation 0.
