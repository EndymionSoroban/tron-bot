# Code Review & Findings

After a thorough review of the repository, I have identified several major architectural and logical issues that affect the training quality, performance, and correctness of the Tron AI.

## 1. CRITICAL: Hardcoded State Perspectives (The "Blind Player 2" Bug)
**Files Affected:** `tron_env.py`, `train_self_play.py`, `train_genetic.py`, `play_vs_ai.py`
**Description:** 
The environment's `get_state()` method is hardcoded to extract feature distances, vector coordinates, and grid layers exclusively from the perspective of `Player 1`. 
When Player 2 (the AI in `play_vs_ai.py`, or the second agent in `train_self_play.py`) gets its state, it is actually receiving Player 1's state. 
This means Player 2 has **no idea** when it is about to hit a wall, because its `dist_straight` feature is actually measuring Player 1's distance to a wall! The AI playing as Player 2 is essentially playing blindfolded while receiving sensory data from its opponent.

**Fix:** Refactor `get_state()` to take a `player_id` parameter, and swap the references to `self.player1` and `self.player2` based on whose perspective is being requested. In `env.step()`, Player 1's state can be returned normally, and Player 2's state can be passed back via the `info` dictionary (e.g. `info['player2_state']`).

## 2. Inefficient Q-Learning Target Computation (Bottleneck)
**Files Affected:** `model.py`
**Description:** 
In `QTrainer.train_step()`, the Bellman equation is computed using a Python `for` loop that iterates over the batch:
```python
for idx in range(len(done)):
    if not done[idx]:
        Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
```
This forces PyTorch to perform `BATCH_SIZE` separate forward passes (`self.model(next_state[idx])`) instead of a single batched forward pass (`self.model(next_states)`). This completely defeats the purpose of GPU/CPU vectorization and drastically slows down long-term memory training.

**Fix:** Vectorize the computation:
```python
q_next = self.model(next_state).max(dim=1)[0]
Q_new = reward + self.gamma * q_next * (1 - done_tensor)
```

## 3. Extremely Fast Exploration Decay
**Files Affected:** `agent.py`
**Description:** 
The epsilon-greedy policy uses the formula `self.epsilon = 80 - self.n_games`. 
Since `train.py` runs for 1000 episodes, the agent reaches 0 exploration after just 8% of the training cycle (80 games). Deep Q-Learning typically requires exploration over thousands of games to discover complex strategies (like trapping the opponent). Cutting it off at 80 games almost guarantees the agent will get stuck in a local minimum (e.g., just driving in a circle forever).

**Fix:** Implement a proper decay factor: `self.epsilon = max(MIN_EPSILON, INITIAL_EPSILON * (DECAY_RATE ** self.n_games))`, or simply stretch the linear decay over `NUM_EPISODES * 0.8`.

## 4. PyGame Resource Leaks in Genetic Training
**Files Affected:** `train_genetic.py`, `tron_env.py`
**Description:** 
When `RENDER_EVERY_GENERATION = True`, `evaluate_model` is called with `render=True`. The `TronEnv` initializes PyGame, renders the game, and then `env.close()` calls `pygame.quit()`. 
Repeatedly initializing and quitting the entire PyGame subsystem inside a loop across 50 generations is resource-intensive and can cause window flashing, OS-level window focus stealing, and potential memory leaks on macOS.

**Fix:** Initialize PyGame once globally at the start of the script if rendering is intended, or use `pygame.display.quit()` instead of `pygame.quit()` if you must tear down the window context.

## 5. Genetic Algorithm Output Consistency (Minor)
**Files Affected:** `train_genetic.py`
**Description:**
The print statement outputs `Score=-1.2`. For consistency and better readability, all floating point output metrics should format to `.1f` to prevent noisy terminal outputs.
