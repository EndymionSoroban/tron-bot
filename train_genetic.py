import torch
import numpy as np
import random
import copy
import os
from tron_env import TronEnv
from agent import TronAgent
from model import LinearQNet

# Genetic Algorithm settings
POPULATION_SIZE = 20  # X: number of models per generation
NUM_GENERATIONS = 50  # Number of generations to run
ELITISM_COUNT = 4  # Y: number of top models to keep as parents
MUTATION_RATE = 0.1  # Probability of mutating a weight
MUTATION_STRENGTH = 0.2  # How much to mutate
GAMES_PER_MODEL = 10  # Games to evaluate each model
STATE_TYPE = 'features'
MODEL_TYPE = 'linear'
RENDER_BEST_RUN = True  # Show the best run from final generation

# Output directories
GENETIC_RUNS_DIR = 'genetic_runs'  # Directory for generation checkpoints
GENETIC_WINNER_DIR = 'genetic_winner'  # Directory for final winner


def crossover(parent1, parent2):
    """Create child by mixing weights from two parents"""
    child = copy.deepcopy(parent1)
    
    for child_param, param1, param2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
        # Randomly choose weights from either parent
        mask = torch.rand_like(child_param) > 0.5
        child_param.data = torch.where(mask, param1.data, param2.data)
    
    return child


def mutate(model, rate=MUTATION_RATE, strength=MUTATION_STRENGTH):
    """Randomly mutate model weights"""
    for param in model.parameters():
        if random.random() < rate:
            noise = torch.randn_like(param) * strength
            param.data += noise


def evaluate_model(model, opponent_model=None, num_games=GAMES_PER_MODEL, render=False):
    """Evaluate a model's fitness by playing games against another model or heuristic"""
    env = TronEnv(render=render)
    agent = TronAgent(state_type=STATE_TYPE, model_type=MODEL_TYPE)
    agent.model = model
    agent.model.eval()  # Set to evaluation mode
    
    # Setup opponent
    if opponent_model is not None:
        opponent_agent = TronAgent(state_type=STATE_TYPE, model_type=MODEL_TYPE)
        opponent_agent.model = opponent_model
        opponent_agent.model.eval()
    else:
        opponent_agent = None
    
    total_score = 0
    wins = 0
    best_score = float('-inf')
    clock = None
    if render:
        import pygame
        clock = pygame.time.Clock()
    
    for _ in range(num_games):
        state_dict = env.reset()
        state = agent.get_state(state_dict)
        done = False
        episode_score = 0
        
        while not done:
            # Handle pygame events if rendering
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return total_score / num_games, wins / num_games
            
            # Get action from model
            action = agent.get_action(state)
            action_idx = action.index(1)
            
            # Get action from opponent
            if opponent_agent is not None:
                opponent_state = opponent_agent.get_state(state_dict)
                opponent_action = opponent_agent.get_action(opponent_state)
                opponent_action_idx = opponent_action.index(1)
            else:
                opponent_action_idx = None
            
            state_dict, reward, done, info = env.step(action_idx, opponent_action_idx)
            state = agent.get_state(state_dict)
            episode_score += reward
            
            if render:
                env._render()
                clock.tick(60)
        
        total_score += reward
        if info['winner'] == 'player1':
            wins += 1
            if episode_score > best_score:
                best_score = episode_score
    
    env.close()
    return total_score / num_games, wins / num_games


def create_random_model():
    """Create a new random model"""
    if STATE_TYPE == 'features':
        model = LinearQNet(17, 256, 3)
    elif STATE_TYPE == 'vector':
        model = LinearQNet(8, 256, 3)
    else:
        raise ValueError(f"Unsupported state type: {STATE_TYPE}")
    return model


def tournament_selection(population, fitness_scores, tournament_size=3):
    """Select a parent using tournament selection"""
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitness)]
    return population[winner_idx]


def genetic_algorithm():
    """Run genetic algorithm training"""
    print(f"Starting Genetic Algorithm Training")
    print(f"Population Size: {POPULATION_SIZE}")
    print(f"Generations: {NUM_GENERATIONS}")
    print(f"Elitism: Keep top {ELITISM_COUNT} models")
    print(f"Mutation Rate: {MUTATION_RATE}")
    print("-" * 50)
    
    # Initialize population
    population = [create_random_model() for _ in range(POPULATION_SIZE)]
    
    for generation in range(NUM_GENERATIONS):
        print(f"\nGeneration {generation + 1}/{NUM_GENERATIONS}")
        
        # Evaluate fitness for all models using tournament style
        fitness_scores = []
        win_rates = []
        
        for i, model in enumerate(population):
            # Each model plays against a random subset of other models
            total_score = 0
            total_wins = 0
            total_games = 0
            
            # Sample opponents (play against random subset of population)
            num_opponents = min(5, len(population) - 1)
            opponent_indices = random.sample([j for j in range(len(population)) if j != i], num_opponents)
            
            for opp_idx in opponent_indices:
                opponent = population[opp_idx]
                games_vs_opp = max(2, GAMES_PER_MODEL // num_opponents)
                
                # Play as player 1
                score1, wins1 = evaluate_model(model, opponent, games_vs_opp)
                total_score += score1
                total_wins += wins1 * games_vs_opp
                total_games += games_vs_opp
                
                # Play as player 2 (swap roles)
                score2, wins2 = evaluate_model(opponent, model, games_vs_opp)
                total_score -= score2  # Negative because we're tracking from player 1 perspective
                total_wins += (games_vs_opp - wins2)  # Opponent's losses are our wins
                total_games += games_vs_opp
            
            avg_score = total_score / total_games if total_games > 0 else 0
            win_rate = total_wins / total_games if total_games > 0 else 0
            
            fitness_scores.append(avg_score)
            win_rates.append(win_rate)
            print(f"  Model {i}: Score={avg_score:.1f}, Win Rate={win_rate:.1%}")
        
        # Sort by fitness
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices]
        fitness_scores = [fitness_scores[i] for i in sorted_indices]
        win_rates = [win_rates[i] for i in sorted_indices]
        
        # Print generation stats
        print(f"  Best Score: {fitness_scores[0]:.1f}")
        print(f"  Best Win Rate: {win_rates[0]:.1%}")
        print(f"  Avg Score: {np.mean(fitness_scores):.1f}")
        
        # Save best model
        best_model = population[0]
        torch.save(best_model.state_dict(), f'genetic_best_gen_{generation}.pth')
        
        # Create next generation
        new_population = []
        
        # Elitism: keep top models
        for i in range(ELITISM_COUNT):
            new_population.append(copy.deepcopy(population[i]))
        
        # Create rest through crossover and mutation
        while len(new_population) < POPULATION_SIZE:
            # Select parents
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutate
            mutate(child)
            
            new_population.append(child)
        
        population = new_population
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("Training Complete!")
    print("Evaluating final population...")
    
    final_fitness = []
    for i, model in enumerate(population):
        # Evaluate against random opponents
        total_score = 0
        total_wins = 0
        total_games = 0
        
        num_opponents = min(5, len(population) - 1)
        opponent_indices = random.sample([j for j in range(len(population)) if j != i], num_opponents)
        
        for opp_idx in opponent_indices:
            opponent = population[opp_idx]
            games_vs_opp = max(3, (GAMES_PER_MODEL * 2) // num_opponents)
            
            score1, wins1 = evaluate_model(model, opponent, games_vs_opp)
            total_score += score1
            total_wins += wins1 * games_vs_opp
            total_games += games_vs_opp
            
            score2, wins2 = evaluate_model(opponent, model, games_vs_opp)
            total_score -= score2
            total_wins += (games_vs_opp - wins2)
            total_games += games_vs_opp
        
        avg_score = total_score / total_games if total_games > 0 else 0
        win_rate = total_wins / total_games if total_games > 0 else 0
        
        final_fitness.append(avg_score)
        print(f"  Final Model {i}: Score={avg_score:.1f}, Win Rate={win_rate:.1%}")
    
    best_idx = np.argmax(final_fitness)
    torch.save(population[best_idx].state_dict(), 'genetic_final_best.pth')
    print(f"\nBest model saved as genetic_final_best.pth")
    
    # Render best run from final generation (play against second best)
    if RENDER_BEST_RUN:
        print("\nRendering best run from final generation...")
        print("Best model vs 2nd best model. Close the window to exit.")
        second_best_idx = np.argsort(final_fitness)[-2]
        evaluate_model(population[best_idx], population[second_best_idx], num_games=1, render=True)


if __name__ == "__main__":
    genetic_algorithm()
