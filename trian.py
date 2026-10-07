import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from othello_network import GameNetworkSimple

# Heuristic weights for board positions (corners, edges, center)
WEIGHTS = np.array([
    [100, -20,  10,   5,   5,  10, -20, 100],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [ 10,  -2,   5,   1,   1,   5,  -2,  10],
    [  5,  -2,   1,   0,   0,   1,  -2,   5],
    [  5,  -2,   1,   0,   0,   1,  -2,   5],
    [ 10,  -2,   5,   1,   1,   5,  -2,  10],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [100, -20,  10,   5,   5,  10, -20, 100]
])

# Normalize weights to (0 to 1 scale)
WEIGHTS_MIN = WEIGHTS.min()
WEIGHTS_MAX = WEIGHTS.max()
WEIGHTS_NORMALIZED = (WEIGHTS - WEIGHTS_MIN) / (WEIGHTS_MAX - WEIGHTS_MIN)

def apply_heuristic_to_policies(policies):
    """Apply heuristic weighting to ground truth policies."""
    modified_policies = []
    for policy in policies:
        policy_board = policy.reshape(8, 8)
        weighted_policy = policy_board * WEIGHTS_NORMALIZED
        weighted_flat = weighted_policy.flatten()
        
        policy_sum = np.sum(weighted_flat)
        if policy_sum > 1e-8:
            normalized_policy = weighted_flat / policy_sum
        else:
            normalized_policy = np.ones(64) / 64
        modified_policies.append(normalized_policy)
    return np.array(modified_policies)

def load_single_dataset(file_path):
    """Load the generated NPZ file directly."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return None, None, None
    
    print(f"Loading dataset from {file_path}...")
    data = np.load(file_path)
    return data['states'], data['policies'], data['values']

def plot_training_history(history, save_path="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Policy Loss
    axes[0].plot(history.history['policy_loss'], label='Train Policy')
    axes[0].plot(history.history['val_policy_loss'], label='Val Policy')
    axes[0].set_title('Policy Loss')
    axes[0].legend()
    # Value Loss
    axes[1].plot(history.history['value_loss'], label='Train Value')
    axes[1].plot(history.history['val_value_loss'], label='Val Value')
    axes[1].set_title('Value Loss')
    axes[1].legend()
    plt.savefig(save_path)
    plt.close()

def train_network(states, policies, values):
    print("\n--- Starting Training Process ---")
    
    # Apply heuristics as per your original logic
    # policies = apply_heuristic_to_policies(policies)
    
    # Create Network (Ensuring action_size=64)
    network = GameNetworkSimple(action_size=64, learning_rate=0.001)
    
    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    
    # Fit Model
    history = network.model.fit(
        states,
        {"policy": policies, "value": values},
        epochs=20,
        batch_size=512,
        validation_split=0.1,
        shuffle=True,
        callbacks=[reduce_lr, early_stop]
    )

    network.model.save("othello_final_pro_model_v3.keras")
    print("\nModel saved as 'othello_final_pro_model_v3.keras'")
    return history

def main():
    # DIRECTLY LOAD YOUR NEW KAGGLE DATA
    data_file = "othello_full_1.5M.npz"
    
    states, policies, values = load_single_dataset(data_file)
    
    if states is not None:
        history = train_network(states, policies, values)
        plot_training_history(history)
        print("Training Completed Successfully!")

if __name__ == "__main__":
    main()