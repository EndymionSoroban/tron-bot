import matplotlib.pyplot as plt
from tron_env import TronEnv
from agent import TronAgent

# Training settings
NUM_EPISODES = 1000
STATE_TYPE = 'features'  # 'vector', 'features', or 'grid'
MODEL_TYPE = 'linear'  # 'linear' or 'conv' (use 'conv' with grid state)
RENDER = True  # Set to True to visualize training, False for headless
FRAME_SKIP = 5  # Skip N frames between renders to speed up visualization
TRAIN_FREQUENCY = 1  # Train every N steps (higher = faster, 1 = train every step)


def plot_scores(p1_scores, p2_scores, mean_scores):
    """Plot training progress for both agents"""
    plt.figure(figsize=(10, 6))
    plt.plot(p1_scores, label='Player 1 Score', alpha=0.4, color='cyan')
    plt.plot(p2_scores, label='Player 2 Score', alpha=0.4, color='magenta')
    plt.plot(mean_scores, label='Mean Score', linewidth=2, color='white')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Self-Play Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('self_play_progress.png')
    plt.close()


def train_self_play():
    """Train two agents against each other"""
    # Initialize environment
    env = TronEnv(render=RENDER)
    clock = None
    if RENDER:
        import pygame
        clock = pygame.time.Clock()
    
    # Initialize two agents
    agent1 = TronAgent(state_type=STATE_TYPE, model_type=MODEL_TYPE)
    agent2 = TronAgent(state_type=STATE_TYPE, model_type=MODEL_TYPE)
    
    # Training metrics
    p1_score_history = []
    p2_score_history = []
    mean_score_history = []
    p1_total_score = 0
    p2_total_score = 0
    p1_wins = 0
    p2_wins = 0
    draws = 0
    
    print(f"Starting self-play training with {STATE_TYPE} state representation...")
    print(f"Model type: {MODEL_TYPE}")
    print(f"Episodes: {NUM_EPISODES}")
    print("-" * 50)
    
    for episode in range(NUM_EPISODES):
        # Reset environment
        state_dict = env.reset()
        state1 = agent1.get_state(state_dict)
        state2 = agent2.get_state(state_dict)
        
        done = False
        episode_reward1 = 0
        episode_reward2 = 0
        step_count = 0
        
        while not done:
            # Handle pygame events if rendering
            if RENDER:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            env.close()
                            return
            
            # Get actions from both agents
            action1 = agent1.get_action(state1)
            action2 = agent2.get_action(state2)
            
            action1_idx = action1.index(1)
            action2_idx = action2.index(1)
            
            # Execute both actions
            next_state_dict, reward, done, info = env.step(action1_idx, action2_idx)
            next_state1 = agent1.get_state(next_state_dict)
            next_state2 = agent2.get_state(next_state_dict)
            
            # Calculate rewards for both agents (opposite for zero-sum game)
            reward1 = reward
            reward2 = -reward  # Zero-sum: one's gain is other's loss
            
            # Train short memory for both agents (only every N steps for speed)
            if step_count % TRAIN_FREQUENCY == 0:
                agent1.train_short_memory(state1, action1, reward1, next_state1, done)
                agent2.train_short_memory(state2, action2, reward2, next_state2, done)
            
            # Remember experiences for both agents
            agent1.remember(state1, action1, reward1, next_state1, done)
            agent2.remember(state2, action2, reward2, next_state2, done)
            
            # Update states
            state1 = next_state1
            state2 = next_state2
            
            episode_reward1 += reward1
            episode_reward2 += reward2
            
            # Render with frame skipping
            if RENDER and step_count % FRAME_SKIP == 0:
                env._render()
                if clock:
                    clock.tick(60)  # 60 FPS for visualization
            
            step_count += 1
        
        # Train long memory for both agents after episode
        agent1.train_long_memory()
        agent2.train_long_memory()
        
        # Update metrics
        agent1.n_games += 1
        agent2.n_games += 1
        p1_total_score += episode_reward1
        p2_total_score += episode_reward2
        mean_score = (p1_total_score + p2_total_score) / (agent1.n_games + agent2.n_games)
        
        # Track wins/losses
        if info['winner'] == 'player1':
            p1_wins += 1
        elif info['winner'] == 'player2':
            p2_wins += 1
        else:
            draws += 1
        
        # Save best models
        if episode_reward1 > agent1.record if hasattr(agent1, 'record') else False:
            agent1.record = episode_reward1
            agent1.save_model('tron_dqn_p1_best.pth')
        
        if episode_reward2 > agent2.record if hasattr(agent2, 'record') else False:
            agent2.record = episode_reward2
            agent2.save_model('tron_dqn_p2_best.pth')
        
        # Save checkpoints every 100 episodes
        if episode % 100 == 0:
            agent1.save_model(f'tron_dqn_p1_episode_{episode}.pth')
            agent2.save_model(f'tron_dqn_p2_episode_{episode}.pth')
        
        # Logging
        p1_score_history.append(episode_reward1)
        p2_score_history.append(episode_reward2)
        mean_score_history.append(mean_score)
        
        if episode % 10 == 0:
            p1_win_rate = p1_wins / (p1_wins + p2_wins + draws) * 100 if (p1_wins + p2_wins + draws) > 0 else 0
            print(f"Episode {episode}/{NUM_EPISODES} | "
                  f"P1 Score: {episode_reward1:.1f} | "
                  f"P2 Score: {episode_reward2:.1f} | "
                  f"Mean: {mean_score:.1f} | "
                  f"P1 Win Rate: {p1_win_rate:.1f}% | "
                  f"W/L/D: {p1_wins}/{p2_wins}/{draws}")
        
        # Plot progress every 50 episodes
        if episode % 50 == 0:
            plot_scores(p1_score_history, p2_score_history, mean_score_history)
    
    # Save final models
    agent1.save_model('tron_dqn_p1_final.pth')
    agent2.save_model('tron_dqn_p2_final.pth')
    
    # Final statistics
    print("\n" + "=" * 50)
    print("Self-Play Training Complete!")
    print(f"Total Episodes: {NUM_EPISODES}")
    print(f"Final Mean Score: {mean_score:.1f}")
    print(f"Player 1 Wins: {p1_wins} ({p1_wins/NUM_EPISODES*100:.1f}%)")
    print(f"Player 2 Wins: {p2_wins} ({p2_wins/NUM_EPISODES*100:.1f}%)")
    print(f"Draws: {draws} ({draws/NUM_EPISODES*100:.1f}%)")
    print("=" * 50)
    
    # Final plot
    plot_scores(p1_score_history, p2_score_history, mean_score_history)
    print("Training progress saved to 'self_play_progress.png'")
    
    env.close()


if __name__ == "__main__":
    train_self_play()
