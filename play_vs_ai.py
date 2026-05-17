import pygame
import sys
from tron_env import TronEnv
from agent import TronAgent

# Game settings
STATE_TYPE = 'features'  # Must match the model you want to load
MODEL_TYPE = 'linear'  # Must match the model you want to load
MODEL_FILE = '/Users/endymion/Documents/Coding projects/ML/tron-bot/genetic_runs/genetic_best_gen_127.pth'  # Model file to load


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
    both_lost = 0
    clock = pygame.time.Clock()
    human_action = None  # Store human's turn decision
    
    while True:
        # Reset environment
        state_dict1 = env.reset()
        state_dict2 = env.get_state(player_id=2)
        state = agent.get_state(state_dict2)
        
        done = False
        game_over = False
        human_action = None
        
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
                        # Human controls - queue turn action
                        if event.key == pygame.K_a:
                            human_action = 1  # Turn left
                        elif event.key == pygame.K_d:
                            human_action = 2  # Turn right
            
            if not game_over:
                # AI makes decision
                action = agent.get_action(state)
                action_idx = action.index(1)
                
                # Execute step with human action (or None if no input) and AI action
                state_dict1, reward, done, info = env.step(human_action, action_idx)
                state = agent.get_state(info['player2_state'])
                
                # Reset human action after using it
                human_action = None
                
                # Render
                env._render()
                
                # Control game speed (slow down for human playability)
                clock.tick(30)
            
            # If game over, show result
            if done and not game_over:
                game_over = True
                
                if info['winner'] == 'player1':
                    human_wins += 1
                    result = "YOU WIN!"
                elif info['winner'] == 'player2':
                    ai_wins += 1
                    result = "AI WINS!"
                else:
                    both_lost += 1
                    result = "BOTH LOST!"
                
                print(f"{result} | Score: {human_wins} - {ai_wins} - {both_lost}")
                print("Press SPACE to play again, ESC to quit")


if __name__ == "__main__":
    play_vs_ai()
