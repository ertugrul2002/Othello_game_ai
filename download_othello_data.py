import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def parse_game_moves(moves_str):
    """
    Convert a move string (e.g., 'f5d6...') into (row, col) coordinates.
    """
    moves = []
    if not isinstance(moves_str, str): 
        return []
    
    for i in range(0, len(moves_str), 2):
        if i + 1 < len(moves_str):
            try:
                col = ord(moves_str[i].lower()) - ord('a') # a-h -> 0-7
                row = int(moves_str[i+1]) - 1              # 1-8 -> 0-7
                if 0 <= row < 8 and 0 <= col < 8:
                    moves.append((row, col))
            except: 
                continue
    return moves

def initialize_board():
    """Initialize the standard Othello starting position."""
    board = np.zeros((8, 8), dtype=np.int8)
    board[3, 3], board[4, 4] = 1, 1 # White pieces
    board[3, 4], board[4, 3] = 2, 2 # Black pieces
    return board

def flip_pieces(board, row, col, player):
    """Update the board by flipping opponent pieces according to Othello rules."""
    opponent = 3 - player
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dr, dc in directions:
        to_flip = []
        nr, nc = row + dr, col + dc
        while 0 <= nr < 8 and 0 <= nc < 8 and board[nr, nc] == opponent:
            to_flip.append((nr, nc))
            nr, nc = nr + dr, nc + dc
        if 0 <= nr < 8 and 0 <= nc < 8 and board[nr, nc] == player:
            for fr, fc in to_flip: 
                board[fr, fc] = player
    return board

def board_to_features(board, current_player):
    """
    Convert board to (8, 8, 3) feature format.
    Channel 2 is now filled correctly using np.full.
    """
    opponent = 3 - current_player
    features = np.zeros((8, 8, 3), dtype=np.float32)
    features[:, :, 0] = (board == current_player).astype(np.float32)
    features[:, :, 1] = (board == opponent).astype(np.float32)
    # Fixed: Use np.full to create a plane of the same value
    features[:, :, 2] = np.full((8, 8), current_player - 1, dtype=np.float32) 
    return features

def process_full_csv(csv_path, output_file="othello_full_1.5M.npz"):
    print(f"--- Processing ALL samples from: {csv_path} ---")
    
    try:
        df = pd.read_csv(csv_path)
        move_col = 'game_moves'
        # We removed the num_samples limit to get everything
        total_games = len(df)
        print(f"Total games to process: {total_games}")
    except Exception as e:
        print(f"Error: {e}")
        return

    all_states, all_policies, all_values = [], [], []
    
    # Progress bar based on games
    for _, row_data in tqdm(df.iterrows(), total=total_games, desc="Extracting Samples"):
        moves_str = row_data[move_col]
        winner_val = row_data.get('winner', 0) 
        moves_list = parse_game_moves(moves_str)
        
        if not moves_list: continue

        board = initialize_board()
        player = 2 # Black starts

        for i, (r, c) in enumerate(moves_list):
            # 1. State
            all_states.append(board_to_features(board, player))
            
            # 2. Policy (Raw move - NO Heuristics here)
            policy = np.zeros(64, dtype=np.float32)
            policy[r * 8 + c] = 1.0
            all_policies.append(policy)
            
            # 3. Value
            val = 0.0
            if winner_val != 0:
                val = 1.0 if winner_val == player else -1.0
            all_values.append(np.array([val], dtype=np.float32))

            board = flip_pieces(board, r, c, player)
            player = 3 - player
    
    print(f"--- Compressing {len(all_states)} samples (This may take a while) ---")
    np.savez_compressed(output_file, 
                        states=np.array(all_states, dtype=np.float16), # Use float16 to save RAM
                        policies=np.array(all_policies, dtype=np.float16), 
                        values=np.array(all_values, dtype=np.float16).squeeze())
    print(f"Done! Saved to {output_file}")

if __name__ == "__main__":
    process_full_csv(r"data\archive\othello_dataset.csv")
