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


def load_dataset(path):
    """
    Load a previously saved dataset.
    
    Args:
        path: Path to the .npz file
        
    Returns:
        Dictionary with boards, policies, values, players
    """
    data = np.load(path)
    return {
        'boards': data['boards'],
        'policies': data['policies'],
        'values': data['values'],
        'players': data['players']
    }


def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(description='Generate MCTS self-play training data for PUCT')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of self-play games to generate (default: 100)')
    parser.add_argument('--simulations', type=int, default=1000,
                       help='Number of MCTS simulations per move (default: 1000)')
    parser.add_argument('--exploration', type=float, default=1.41,
                       help='UCB1 exploration weight (default: 1.41)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Simulation temperature (default: 1.0)')
    parser.add_argument('--output', type=str, default='data/training_data.npz',
                       help='Output path for dataset (default: data/training_data.npz)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Create data generator
    generator = DataGenerator(
        num_simulations=args.simulations,
        exploration_weight=args.exploration,
        simulation_temperature=args.temperature,
        verbose=not args.quiet
    )
    
    # Generate dataset
    dataset = generator.generate_dataset(num_games=args.games)
    
    # Save dataset
    generator.save_dataset(dataset, args.output)
    
    print("\n✅ Data generation complete!")


if __name__ == "__main__":
    # Example: Generate 100 games with 1000 simulations each
    # This will create high-quality training data
    
    # Option 1: Use command line
    # python generate_training_data.py --games 100 --simulations 1000 --output data/training_data.npz
    
    # Option 2: Use directly in code
    generator = DataGenerator(
        num_simulations=200,      # More simulations = better quality but slower
        exploration_weight=1.41,   # Standard UCB1 exploration
        simulation_temperature=1.0, # Balanced exploration/exploitation
        verbose=False               # Show progress
    )
    
    # Generate dataset
    dataset = generator.generate_dataset(num_games=100)  # Start with 10 games for testing
    
    # Save dataset
    generator.save_dataset(dataset, 'data/training_data200_100g.npz')
    
    # Load and verify
    print("\n=== Verification ===")
    loaded_data = load_dataset('data/training_data200_100g.npz')
    print(f"Loaded {len(loaded_data['boards'])} examples")
    print(f"First board shape: {loaded_data['boards'][0].shape}")
    print(f"First policy shape: {loaded_data['policies'][0].shape}")
    print(f"First policy sum: {np.sum(loaded_data['policies'][0]):.6f} (should be 1.0)")
    print(f"First value: {loaded_data['values'][0]} (should be -1, 0, or 1)")
