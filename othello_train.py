import numpy as np
import os
from othello_network_improved import network

def prepare_input(boards, players):
    """
    Convert (N, 8, 8) boards and player info into (N, 8, 8, 3) for the CNN
    Channel 0: Current player's pieces
    Channel 1: Opponent's pieces
    Channel 2: All 1s if it's Player 1's turn, 0s if Player 2
    """
    num_samples = boards.shape[0]
    x_input = np.zeros((num_samples, 8, 8, 3), dtype=np.float32)
    
    for i in range(num_samples):
        p = players[i]
        opp = 2 if p == 1 else 1
        x_input[i, :, :, 0] = (boards[i] == p).astype(np.float32)
        x_input[i, :, :, 1] = (boards[i] == opp).astype(np.float32)
        x_input[i, :, :, 2] = 1.0 if p == 1 else 0.0
        
    return x_input

def train():
    # 1. Initialize your network
    # Learning rate 0.001 is standard for Adam
    othello_nn = network(action_size=64, learning_rate=0.001)

    # 2. Load the data from your specific file
    data_path = 'training_data200_100g.npz'
    print(f"Loading data from {data_path}...")
    
    with np.load(data_path) as data:
        raw_boards = data['boards']    # Shape: (N, 8, 8)
        policies = data['policies']    # Shape: (N, 8, 8)
        values = data['values']        # Shape: (N,)
        players = data['players']      # Shape: (N,)

    # 3. Preprocess data to match CNN input (8, 8, 3)
    print("Preprocessing boards for the network...")
    states = prepare_input(raw_boards, players)
    
    # Flatten policies from (8,8) to (64,) if they aren't already
    policies = policies.reshape(-1, 64)

    # 4. Augmentation (Rotations and Flips)
    # Using the function you wrote in your network class
    print("Augmenting data...")
    x_train, y_policy, y_value = othello_nn.augment_data(states, policies, values)

    # 5. Training
    print(f"Starting training on {len(x_train)} samples...")
    othello_nn.model.fit(
        x_train, 
        {'policy': y_policy, 'value': y_value},
        epochs=20,
        batch_size=64,
        validation_split=0.1,
        shuffle=True
    )

    # 6. Save the results
    othello_nn.model.save("othello_model_v100.keras")
    print("Training finished successfully! Model saved as 'othello_model_v100.keras'")

if __name__ == "__main__":
    train()