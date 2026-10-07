"""
MCTS Self-Play Data Generation for PUCT Training

This script generates high-quality training data by having MCTS play against itself.
The data includes:
- Board states (8x8 arrays)
- Policy targets (move probability distributions from MCTS visit counts)
- Value targets (game outcomes: +1 for win, -1 for loss, 0 for draw)

Usage:
    python generate_training_data.py --games 100 --simulations 1000 --output data/training_data.npz
"""

import numpy as np
import argparse
import os
from tqdm import tqdm
from othello import Othello
from othello_mcts_improved import MCTS
from multiprocessing import Pool, cpu_count

class DataGenerator:
    """
    Generates training data from MCTS self-play games.
    """
    
    def __init__(self, num_simulations=1000, exploration_weight=1.41, 
                 simulation_temperature=1.0, verbose=True):
        """
        Initialize the data generator.
        
        Args:
            num_simulations: Number of MCTS simulations per move
            exploration_weight: UCB1 exploration parameter
            simulation_temperature: Temperature for simulation softmax
            verbose: Whether to print progress information
        """
        self.num_simulations = num_simulations
        self.mcts = MCTS(exploration_weight=exploration_weight,
                        simulation_temperature=simulation_temperature)
        self.verbose = verbose
    
    def play_game(self, game_id=0):
        """
        Play a single self-play game and collect training examples.
        
        Args:
            game_id: Game identifier for logging
            
        Returns:
            List of training examples, each containing:
            - board: 8x8 numpy array representing the board state
            - policy: 8x8 numpy array with move probabilities (0 for illegal moves)
            - value: +1 if current player won, -1 if lost, 0 if draw
            - player: which player's turn (1 for Black, 2 for White)
        """
        game = Othello()
        training_examples = []
        move_count = 0
        
        if self.verbose:
            print(f"\n=== Game {game_id} ===")
        
        while True:
            is_over, winner = game.status()
            
            if is_over:
                # Game ended, assign values to all examples
                if self.verbose:
                    black_score = sum(row.count(game.BLACK) for row in game.board)
                    white_score = sum(row.count(game.WHITE) for row in game.board)
                    print(f"Game Over! Black: {black_score}, White: {white_score}")
                    if winner == Othello.BLACK:
                        print("Winner: BLACK")
                    elif winner == Othello.WHITE:
                        print("Winner: WHITE")
                    else:
                        print("DRAW")
                
                # Assign value from each player's perspective
                for example in training_examples:
                    if winner == 0:
                        # Draw
                        example['value'] = 0.0
                    elif winner == example['player']:
                        # This player won
                        example['value'] = 1.0
                    else:
                        # This player lost
                        example['value'] = -1.0
                
                break
            
            # Get legal moves
            legal_moves = game.legal_moves()
            
            if not legal_moves:
                # No legal moves, pass turn
                if self.verbose:
                    player_name = "Black" if game.current_player == Othello.BLACK else "White"
                    print(f"Move {move_count}: {player_name} passes (no legal moves)")
                game.make_move(None)
                continue
            
            # Run MCTS to get move statistics
            stats = self.mcts.get_move_statistics(game, num_simulations=self.num_simulations)
            
            # Create policy array (8x8) with probabilities for legal moves
            policy = np.zeros((8, 8), dtype=np.float32)
            for move, move_stats in stats.items():
                row, col = move
                policy[row][col] = move_stats['visit_probability']
            
            # Verify policy is valid
            policy_sum = np.sum(policy)
            assert abs(policy_sum - 1.0) < 1e-5, f"Policy sum is {policy_sum}, should be 1.0"
            
            # Store training example
            training_example = {
                'board': np.array(game.board, dtype=np.int8),  # 8x8 board state
                'policy': policy,                               # 8x8 policy distribution
                'player': game.current_player,                  # 1 or 2
                'value': None  # Will be filled after game ends
            }
            training_examples.append(training_example)
            
            # Select best move (highest visit count)
            best_move = max(stats.items(), key=lambda x: x[1]['visits'])[0]
            
            if self.verbose:
                player_name = "Black" if game.current_player == Othello.BLACK else "White"
                print(f"Move {move_count}: {player_name} plays {best_move} "
                      f"(visits: {stats[best_move]['visits']}, "
                      f"win_rate: {stats[best_move]['win_rate']:.3f})")
            
            # Make the move
            game.make_move(best_move)
            move_count += 1
        
        if self.verbose:
            print(f"Collected {len(training_examples)} training examples from this game\n")
        
        return training_examples
    
    def generate_dataset(self, num_games=100):
        """
        Generate a dataset by playing multiple self-play games.
        
        Args:
            num_games: Number of games to play
            
        Returns:
            Dictionary containing:
            - boards: (N, 8, 8) array of board states
            - policies: (N, 8, 8) array of policy targets
            - values: (N,) array of value targets
            - players: (N,) array of player indicators
        """
        all_examples = []
        
        print(f"Generating training data from {num_games} self-play games...")
        print(f"MCTS settings: {self.num_simulations} simulations per move")
        print(f"Exploration weight: {self.mcts.exploration_weight}")
        print(f"Simulation temperature: {self.mcts.simulation_temperature}\n")
        
        # Play games with progress bar
        for game_id in tqdm(range(num_games), desc="Playing games"):
            game_examples = self.play_game(game_id=game_id)
            all_examples.extend(game_examples)
        
        # Convert to numpy arrays
        boards = np.array([ex['board'] for ex in all_examples], dtype=np.int8)
        policies = np.array([ex['policy'] for ex in all_examples], dtype=np.float32)
        values = np.array([ex['value'] for ex in all_examples], dtype=np.float32)
        players = np.array([ex['player'] for ex in all_examples], dtype=np.int8)
        
        dataset = {
            'boards': boards,
            'policies': policies,
            'values': values,
            'players': players
        }
        
        print(f"\n=== Dataset Summary ===")
        print(f"Total examples: {len(boards)}")
        print(f"Board shape: {boards.shape}")
        print(f"Policy shape: {policies.shape}")
        print(f"Value shape: {values.shape}")
        print(f"Value distribution: Win={np.sum(values == 1)}, "
              f"Loss={np.sum(values == -1)}, Draw={np.sum(values == 0)}")
        
        return dataset
    
    def save_dataset(self, dataset, output_path):
        """
        Save dataset to disk in compressed numpy format.
        
        Args:
            dataset: Dictionary with boards, policies, values, players
            output_path: Path to save the dataset (e.g., 'data/training_data.npz')
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as compressed numpy archive
        np.savez_compressed(
            output_path,
            boards=dataset['boards'],
            policies=dataset['policies'],
            values=dataset['values'],
            players=dataset['players']
        )
        
        # Calculate file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"\n=== Dataset Saved ===")
        print(f"Path: {output_path}")
        print(f"Size: {file_size_mb:.2f} MB")
        print(f"\nTo load the dataset:")
        print(f"  data = np.load('{output_path}')")
        print(f"  boards = data['boards']")
        print(f"  policies = data['policies']")
        print(f"  values = data['values']")
        print(f"  players = data['players']")


def run_single_game(args_tuple):
    game_id, simulations, exploration, temperature, output_dir = args_tuple
    generator = DataGenerator(
        num_simulations=simulations,
        exploration_weight=exploration,
        simulation_temperature=temperature,
        verbose=False 
    )
    dataset = generator.generate_dataset(num_games=1)
    temp_path = os.path.join(output_dir, f"game_{game_id:03d}.npz") # 001, 002... لتسهيل الترتيب
    generator.save_dataset(dataset, temp_path)
    return temp_path

def main():
    # إعدادات المعالج لـ Ryzen 9
    total_cores = cpu_count()
    # نترك 2 Cores للنظام عشان ما يعلق الجهاز والباقي للشغل
    recommended_workers = max(1, total_cores - 2) 

    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--simulations', type=int, default=1000)
    parser.add_argument('--output', type=str, default='data/training_data.npz')
    parser.add_argument('--workers', type=int, default=recommended_workers)
    args = parser.parse_args()

    temp_dir = "temp_data"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    tasks = [(i, args.simulations, 1.41, 1.0, temp_dir) for i in range(args.games)]

    print(f"🔥 Ryzen 9 detected with {total_cores} threads.")
    print(f"🚀 Using {args.workers} workers to generate {args.games} games...")

    # استخدام Pool مع توزيع المهام بشكل أسرع
    with Pool(processes=args.workers) as pool:
        pool.map(run_single_game, tasks, chunksize=1)

    # --- التجميع الذكي ---
    print("\n📦 Gathering data and merging...")
    import glob
    all_files = sorted(glob.glob(os.path.join(temp_dir, "*.npz")))
    
    final_boards = []
    final_policies = []
    final_values = []
    final_players = []

    for f in all_files:
        data = np.load(f)
        final_boards.append(data['boards'])
        final_policies.append(data['policies'])
        final_values.append(data['values'])
        final_players.append(data['players'])
        os.remove(f)

    np.savez_compressed(
        args.output,
        boards=np.concatenate(final_boards),
        policies=np.concatenate(final_policies),
        values=np.concatenate(final_values),
        players=np.concatenate(final_players)
    )
    
    if not os.listdir(temp_dir):
        os.rmdir(temp_dir)
        
    print(f"✨ Success! Final dataset size: {len(np.concatenate(final_boards))} states.")

if __name__ == "__main__":
    main()