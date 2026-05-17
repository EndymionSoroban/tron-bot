import matplotlib.pyplot as plt
import os
from tron_env import TronEnv
from agent import TronAgent

# Training settings
NUM_EPISODES = 1000
STATE_TYPE = 'features'  # 'vector', 'features', or 'grid'
MODEL_TYPE = 'linear'  # 'linear' or 'conv' (use 'conv' with grid state)


def plot_scores(scores, mean_scores):
    """Plot training progress"""
    plt.figure(figsize=(10, 6))
    plt.plot(scores, label='Score', alpha=0.6)
    plt.plot(mean_scores, label='Mean Score', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    os.makedirs('runs', exist_ok=True)
    plt.savefig('runs/training_progress.png')
    plt.close()


def train():
    """Main training loop"""
    # Initialize environment (headless for training)
    env = TronEnv(render=False)
    
    # Initialize agent
    agent = TronAgent(state_type=STATE_TYPE, model_type=MODEL_TYPE)
    
    # Training metrics
    score_history = []
    mean_score_history = []
    total_score = 0
    record = 0
    wins = 0
    losses = 0
    draws = 0
    
    print(f"Starting training with {STATE_TYPE} state representation...")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Episodes: {NUM_EPISODES}")
    print("-" * 50)
    
    for episode in range(NUM_EPISODES):
        # Reset environment
        state_dict = env.reset()
        state = agent.get_state(state_dict)
        
        done = False
        episode_reward = 0
        
        while not done:
            # Get action from agent
            action = agent.get_action(state)
            action_idx = action.index(1)
            
            # Execute action (opponent uses simple heuristic)
            next_state_dict, reward, done, info = env.step(action_idx)
            next_state = agent.get_state(next_state_dict)
            
            # Train short memory
            agent.train_short_memory(state, action, reward, next_state, done)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            episode_reward += reward
        
        # Train long memory after episode
        agent.train_long_memory()
        
        # Update metrics
        agent.n_games += 1
        total_score += episode_reward
        mean_score = total_score / agent.n_games
        
        # Track wins/losses
        if info['winner'] == 'player1':
            wins += 1
        elif info['winner'] == 'player2':
            losses += 1
        else:
            draws += 1
        
        # Update record
        if episode_reward > record:
            record = episode_reward
            agent.save_model('tron_dqn_best.pth')
        
        # Save checkpoint every 100 episodes
        if episode % 100 == 0:
            agent.save_model(f'tron_dqn_episode_{episode}.pth')
        
        # Logging
        score_history.append(episode_reward)
        mean_score_history.append(mean_score)
        
        if episode % 10 == 0:
            win_rate = wins / agent.n_games * 100 if agent.n_games > 0 else 0
            print(f"Episode {episode}/{NUM_EPISODES} | "
                  f"Score: {episode_reward:.1f} | "
                  f"Mean: {mean_score:.1f} | "
                  f"Record: {record:.1f} | "
                  f"Win Rate: {win_rate:.1f}% | "
                  f"W/L/D: {wins}/{losses}/{draws}")
        
        # Plot progress every 50 episodes
        if episode % 50 == 0:
            plot_scores(score_history, mean_score_history)
    
    # Save final model
    agent.save_model('tron_dqn_final.pth')
    
    # Final statistics
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Total Episodes: {NUM_EPISODES}")
    print(f"Final Mean Score: {mean_score:.1f}")
    print(f"Best Score: {record:.1f}")
    print(f"Win Rate: {wins / NUM_EPISODES * 100:.1f}%")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print("=" * 50)
    
    # Final plot
    plot_scores(score_history, mean_score_history)
    print("Training progress saved to 'runs/training_progress.png'")
    
    env.close()


if __name__ == "__main__":
    train()
