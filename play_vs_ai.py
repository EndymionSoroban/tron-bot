import pygame
import sys
from tron_env import TronEnv
from agent import TronAgent

# Game settings
STATE_TYPE = 'vector'  # Must match the model you want to load
MODEL_TYPE = 'linear'  # Must match the model you want to load
MODEL_FILE = 'tron_dqn_best.pth'  # Model file to load


def play_vs_ai():
    """Play against trained AI"""
    # Initialize environment with rendering
    env = TronEnv(render=True)
    
    # Initialize agent and load model
    agent = TronAgent(state_type=STATE_TYPE, model_type=MODEL_TYPE)
    
    if not agent.load_model(MODEL_FILE):
        print(f"Error: Could not load model '{MODEL_FILE}'")
        print("Make sure you have trained a model first using train.py")
        env.close()
        sys.exit(1)
    
    print(f"Loaded model: {MODEL_FILE}")
    print("Controls: A/D to turn left/right")
    print("Press SPACE to restart after game over")
    print("Press ESC to quit")
    print("-" * 50)
    
    # Game state
    human_wins = 0
    ai_wins = 0
    draws = 0
    
    while True:
        # Reset environment
        state_dict = env.reset()
        state = agent.get_state(state_dict)
        
        done = False
        game_over = False
        
        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    sys.exit()
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        env.close()
                        sys.exit()
                    
                    if game_over and event.key == pygame.K_SPACE:
                        done = True  # Break inner loop to restart
                    
                    if not game_over:
                        # Human controls
                        if event.key == pygame.K_a:
                            # Human turn left
                            state_dict, reward, done, info = env.step(0, None)
                        elif event.key == pygame.K_d:
                            # Human turn right
                            state_dict, reward, done, info = env.step(1, None)
                        
                        if not done:
                            state = agent.get_state(state_dict)
                            
                            # AI turn
                            action = agent.get_action(state)
                            action_idx = action.index(1)
                            state_dict, reward, done, info = env.step(action_idx, action_idx)
                            state = agent.get_state(state_dict)
            
            # If game over, show result
            if done and not game_over:
                game_over = True
                winner = info['winner']
                
                if winner == 'player1':
                    human_wins += 1
                    result = "YOU WIN!"
                elif winner == 'player2':
                    ai_wins += 1
                    result = "AI WINS!"
                else:
                    draws += 1
                    result = "DRAW!"
                
                print(f"{result} | Score: {human_wins} - {ai_wins} - {draws}")
                print("Press SPACE to play again, ESC to quit")
                
                # Continue rendering to show game over screen
                env._render()
                env.pygame.time.wait(100)  # Small delay to reduce CPU usage


if __name__ == "__main__":
    play_vs_ai()
