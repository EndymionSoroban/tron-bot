import torch
import torch.multiprocessing as mp
import numpy as np
import random
import copy
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from tron_env import TronEnv
from agent import TronAgent
from model import LinearQNet

# Required for macOS multiprocessing with PyTorch
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Genetic Algorithm settings
POPULATION_SIZE = 80  # X: number of models per generation
NUM_GENERATIONS = 300  # Number of generations to run
ELITISM_COUNT = 8  # Y: number of top models to keep as parents
MUTATION_RATE = 0.1  # Probability of mutating a weight
MUTATION_STRENGTH = 0.15  # How much to mutate
GAMES_PER_MODEL = 16  # Games to evaluate each model
STATE_TYPE = 'features'
MODEL_TYPE = 'linear'
RENDER_BEST_RUN = True  # Show the best run from final generation
RENDER_EVERY_GENERATION = False  # Render the best model of each generation against the 2nd best
NUM_WORKERS = 8  # Number of parallel workers for evaluation (set to CPU core count)

# Output directories
GENETIC_RUNS_DIR = 'genetic_runs'  # Directory for generation checkpoints
GENETIC_WINNER_DIR = 'genetic_winner'  # Directory for final winner

# Seeding: provide one or more trained model paths to seed the initial population
# The population will be generated via crossover + boosted mutation from these parents
SEED_MODELS = []  # e.g. ['genetic_winner/genetic_winner_2026-05-17.pth', 'checkpoints/tron_dqn_best.pth']
SEED_MUTATION_STRENGTH = 0.5  # Higher mutation for initial population diversity (normal training uses 0.2)


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


def _evaluate_single_model(args):
    """Worker function: evaluate one model against its assigned opponents and a smart heuristic.
    Accepts serialized state_dicts to avoid pickling issues with multiprocessing."""
    model_idx, model_state, opponent_states, opponent_indices, games_vs_opp = args
    
    # Reconstruct models from state dicts
    model = create_random_model()
    model.load_state_dict(model_state)
    
    total_score = 0
    total_wins = 0
    total_games = 0
    
    # 1. Play against assigned neural network opponents from the population
    for opp_state in opponent_states:
        opponent = create_random_model()
        opponent.load_state_dict(opp_state)
        
        # Play as player 1
        raw_score1, _, raw_wins1, games_played1 = evaluate_model(model, opponent, games_vs_opp)
        total_score += raw_score1
        total_wins += raw_wins1
        total_games += games_played1
        
        # Play as player 2 (swap roles)
        _, raw_score2, raw_opp_wins2, games_played2 = evaluate_model(opponent, model, games_vs_opp)
        total_score += raw_score2
        total_wins += (games_played2 - raw_opp_wins2)
        total_games += games_played2
        
    # 2. Play against the Smart Heuristic opponent for baseline survival testing
    # 2 games as player 1
    h_score1, _, h_wins1, h_games1 = evaluate_model(model, opponent_model=None, num_games=2)
    total_score += h_score1
    total_wins += h_wins1
    total_games += h_games1
    
    # 2 games as player 2
    _, h_score2, h_opp_wins2, h_games2 = evaluate_model(model=None, opponent_model=model, num_games=2)
    total_score += h_score2
    total_wins += (h_games2 - h_opp_wins2)
    total_games += h_games2
    
    avg_score = total_score / total_games if total_games > 0 else 0
    win_rate = total_wins / total_games if total_games > 0 else 0
    
    return model_idx, avg_score, win_rate


def evaluate_model(model, opponent_model=None, num_games=GAMES_PER_MODEL, render=False):
    """Evaluate a model's fitness by playing games against another model or heuristic"""
    env = TronEnv(render=render)
    
    # Setup player 1 agent
    if model is not None:
        agent1 = TronAgent(state_type=STATE_TYPE, model_type=MODEL_TYPE)
        agent1.model = model
        agent1.model.eval()
    else:
        agent1 = None
        
    # Setup player 2 agent
    if opponent_model is not None:
        agent2 = TronAgent(state_type=STATE_TYPE, model_type=MODEL_TYPE)
        agent2.model = opponent_model
        agent2.model.eval()
    else:
        agent2 = None
        
    total_score1 = 0
    total_score2 = 0
    wins = 0
    best_score = float('-inf')
    clock = None
    if render:
        import pygame
        clock = pygame.time.Clock()
        
    for _ in range(num_games):
        state_dict1 = env.reset()
        if agent1 is not None:
            state = agent1.get_state(state_dict1)
        if agent2 is not None:
            state_dict2 = env.get_state(player_id=2)
            opponent_state = agent2.get_state(state_dict2)
            
        done = False
        episode_score1 = 0
        episode_score2 = 0
        
        while not done:
            # Handle pygame events if rendering
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return total_score1, total_score2, wins, num_games
            
            # Action for player 1
            if agent1 is not None:
                action1 = agent1.get_action(state)
                action_idx1 = action1.index(1)
            else:
                action_idx1 = 'heuristic'
                
            # Action for player 2
            if agent2 is not None:
                action2 = agent2.get_action(opponent_state)
                action_idx2 = action2.index(1)
            else:
                action_idx2 = 'heuristic'
                
            state_dict1, reward, done, info = env.step(action_idx1, action_idx2)
            
            if agent1 is not None:
                state = agent1.get_state(state_dict1)
            if agent2 is not None:
                opponent_state = agent2.get_state(info['player2_state'])
                
            episode_score1 += reward
            episode_score2 += info['player2_reward']
            
            if render:
                env._render()
                clock.tick(60)
                
        total_score1 += episode_score1
        total_score2 += episode_score2
        
        if info['winner'] == 'player1':
            wins += 1
            if episode_score1 > best_score:
                best_score = episode_score1
                
    env.close()
    return total_score1, total_score2, wins, num_games


def create_random_model():
    """Create a new random model"""
    if STATE_TYPE == 'features':
        model = LinearQNet(22, 256, 3)
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
    os.makedirs(GENETIC_RUNS_DIR, exist_ok=True)
    os.makedirs(GENETIC_WINNER_DIR, exist_ok=True)
    
    print(f"Starting Genetic Algorithm Training")
    print(f"Population Size: {POPULATION_SIZE}")
    print(f"Generations: {NUM_GENERATIONS}")
    print(f"Elitism: Keep top {ELITISM_COUNT} models")
    print(f"Mutation Rate: {MUTATION_RATE}")
    print("-" * 50)
    
    # Initialize population
    if SEED_MODELS:
        # Load seed models
        seed_models = []
        for path in SEED_MODELS:
            if os.path.exists(path):
                m = create_random_model()
                m.load_state_dict(torch.load(path))
                seed_models.append(m)
                print(f"Loaded seed model: {path}")
            else:
                print(f"Warning: Seed model not found: {path}")
        
        if not seed_models:
            print("No valid seed models found. Starting from random population.")
            population = [create_random_model() for _ in range(POPULATION_SIZE)]
        else:
            population = []
            # Keep exact copies of each seed model
            for m in seed_models:
                population.append(copy.deepcopy(m))
            
            # Fill rest of population via crossover + boosted mutation
            while len(population) < POPULATION_SIZE:
                if len(seed_models) >= 2:
                    # Pick two random parents and crossover
                    p1, p2 = random.sample(seed_models, 2)
                    child = crossover(p1, p2)
                else:
                    # Single parent: just clone it
                    child = copy.deepcopy(seed_models[0])
                
                # Apply boosted mutation for diversity
                mutate(child, rate=1.0, strength=SEED_MUTATION_STRENGTH)
                population.append(child)
            
            print(f"Seeded population: {len(seed_models)} exact parent(s) + {POPULATION_SIZE - len(seed_models)} mutated children")
    else:
        population = [create_random_model() for _ in range(POPULATION_SIZE)]
    
    for generation in range(NUM_GENERATIONS):
        print(f"\nGeneration {generation + 1}/{NUM_GENERATIONS}")
        
        # Evaluate fitness for all models using tournament style (parallelized)
        num_opponents = min(5, len(population) - 1)
        games_vs_opp = max(2, GAMES_PER_MODEL // num_opponents)
        
        # Serialize models and prepare worker tasks
        population_states = [m.state_dict() for m in population]
        tasks = []
        for i in range(len(population)):
            opponent_indices = random.sample([j for j in range(len(population)) if j != i], num_opponents)
            opponent_states = [population_states[j] for j in opponent_indices]
            tasks.append((i, population_states[i], opponent_states, opponent_indices, games_vs_opp))
        
        # Run evaluations in parallel
        fitness_scores = [0.0] * len(population)
        win_rates = [0.0] * len(population)
        
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(executor.map(_evaluate_single_model, tasks))
        
        for model_idx, avg_score, win_rate in results:
            fitness_scores[model_idx] = avg_score
            win_rates[model_idx] = win_rate
            print(f"  Model {model_idx}: Score={avg_score:.1f}, Win Rate={win_rate:.1%}")
        
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
        save_path = os.path.join(GENETIC_RUNS_DIR, f'genetic_best_gen_{generation}.pth')
        torch.save(best_model.state_dict(), save_path)
        
        # Render best run of this generation
        if RENDER_EVERY_GENERATION and len(population) >= 2:
            print(f"  Rendering best model vs 2nd best model for generation {generation + 1}...")
            evaluate_model(population[0], population[1], num_games=1, render=True)
        
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
    
    # Parallelize final evaluation
    num_opponents = min(5, len(population) - 1)
    games_vs_opp = max(3, (GAMES_PER_MODEL * 2) // num_opponents)
    
    population_states = [m.state_dict() for m in population]
    tasks = []
    for i in range(len(population)):
        opponent_indices = random.sample([j for j in range(len(population)) if j != i], num_opponents)
        opponent_states = [population_states[j] for j in opponent_indices]
        tasks.append((i, population_states[i], opponent_states, opponent_indices, games_vs_opp))
    
    final_fitness = [0.0] * len(population)
    final_win_rates = [0.0] * len(population)
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(_evaluate_single_model, tasks))
    
    for model_idx, avg_score, win_rate in results:
        final_fitness[model_idx] = avg_score
        final_win_rates[model_idx] = win_rate
        print(f"  Final Model {model_idx}: Score={avg_score:.1f}, Win Rate={win_rate:.1%}")
    
    # Sort by fitness (descending)
    sorted_indices = np.argsort(final_fitness)[::-1]
    best_score = final_fitness[sorted_indices[0]]
    
    # Create timestamped subdirectory: genetic_winner/2026-05-17_13-54_score182/
    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
    run_dir_name = f"{date_str}_score{best_score:.0f}"
    run_dir = os.path.join(GENETIC_WINNER_DIR, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save top 8 models with placing and win rate
    num_to_save = min(8, len(population))
    print(f"\nSaving top {num_to_save} models to {run_dir}/")
    for rank in range(num_to_save):
        idx = sorted_indices[rank]
        score = final_fitness[idx]
        wr = final_win_rates[idx]
        filename = f"place{rank+1}_score{score:.0f}_wr{wr*100:.0f}.pth"
        save_path = os.path.join(run_dir, filename)
        torch.save(population[idx].state_dict(), save_path)
        print(f"  #{rank+1}: {filename}")
    
    # Render best run from final generation (play against second best)
    if RENDER_BEST_RUN:
        print("\nRendering best run from final generation...")
        print("Best model vs 2nd best model. Close the window to exit.")
        evaluate_model(population[sorted_indices[0]], population[sorted_indices[1]], num_games=1, render=True)


if __name__ == "__main__":
    genetic_algorithm()
