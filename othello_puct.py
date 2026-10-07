import random
import math
import numpy as np
from othello import Othello
from othello_network_improved import GameNetworkSimple
import tensorflow as tf

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


def random_policy(legal_moves):
    """
    Generate a uniform random policy over the given legal moves.
    """
    if not legal_moves:
        return None # No legal moves available
    
    num_moves = len(legal_moves)
    probability = 1.0 / num_moves
    policy_map = {move: probability for move in legal_moves}
    
    return policy_map


# def get_legal_policy_dict(policy_vector, legal_moves, board_size=8):
    
#     legal_probs = []
    
#     for r, c in legal_moves:
#         idx = r * board_size + c
#         prob = policy_vector[idx]
#         legal_probs.append(prob)
#         # print(f"Move: ({r},{c}), Index: {idx}, Prob: {prob}")
    
#     total = sum(legal_probs)

#     if total < 1e-8:
#         if not legal_moves:
#             return {}
#         uniform = 1.0 / len(legal_moves)
#         return {move: uniform for move in legal_moves}
    
#     normalized = [p / total for p in legal_probs]
    
#     return dict(zip(legal_moves, normalized))


def get_legal_policy_dict(policy_vector, legal_moves, board_size=8):
    """
    Extract legal moves from policy vector and apply heuristic weighting.
    
    Process:
    1. Extract probabilities for legal moves
    2. Normalize to get valid probability distribution
    3. Apply heuristic weights (corners, edges, center)
    4. Normalize again to ensure valid probabilities
    
    Args:
        policy_vector: Output from neural network (64 values)
        legal_moves: List of legal moves (tuples)
        board_size: Size of the board (default 8)
    
    Returns:
        Dictionary mapping moves to their weighted probabilities
    """
    
    # Step 1: Extract legal move probabilities
    legal_probs = []
    
    for r, c in legal_moves:
        idx = r * board_size + c
        prob = policy_vector[idx]
        legal_probs.append(prob)
    
    # Step 2: First normalization (get valid probability distribution)
    total = sum(legal_probs)
    
    if total < 1e-8:
        # If all probabilities are near zero, use uniform distribution
        if not legal_moves:
            return {}
        uniform = 1.0 / len(legal_moves)
        return {move: uniform for move in legal_moves}
    
    # Normalize to get probabilities between 0 and 1
    normalized_probs = [p / total for p in legal_probs]
    
    # Step 3: Apply heuristic weights (corners, edges, center)
    weighted_probs = []
    for (r, c), prob in zip(legal_moves, normalized_probs):
        weight = WEIGHTS[r, c]
        # Apply weight (shift to ensure positive values)
        # Weights range from -50 to 100, so we shift by 51 to make them all positive
        shifted_weight = weight + 51  # Now ranges from 1 to 151
        weighted_prob = prob * shifted_weight
        weighted_probs.append(weighted_prob)
    
    # Step 4: Second normalization (ensure valid probabilities)
    weighted_total = sum(weighted_probs)
    
    if weighted_total < 1e-8:
        # Fallback to uniform if weighting resulted in all zeros
        uniform = 1.0 / len(legal_moves)
        return {move: uniform for move in legal_moves}
    
    # Final normalization
    final_probs = [p / weighted_total for p in weighted_probs]
    
    return dict(zip(legal_moves, final_probs))



class PUCTNode:
    """
    Node in the PUCT (Polynomial Upper Confidence Tree) search tree.
    
    Attributes:
        game_state: Current Othello game state
        parent: Parent node
        move: Move that led to this state
        children: List of child nodes
        N: Number of visits to this node
        W: Sum of values from this node
        P: Prior probability from neural network
        is_expanded: Whether this node has been expanded
    """
    
    def __init__(self, game_state, parent=None, move=None, prior=0.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move 
        self.player = game_state.current_player
        self.children = [] 
        self.N = 0  # Visit count
        self.W = 0  # Value sum
        self.P = prior  # Prior probability from policy head
        self.is_expanded = False

    def is_fully_expanded(self) -> bool:
        """Check if all possible moves from this node have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal game state."""
        if self.game_state is None: 
            return True
        is_over, _ = self.game_state.status()
        return is_over
    
    def best_child(self, c_puct: float = 1.41) -> 'PUCTNode':
        """
        Select the best child using PUCT formula.
        
        PUCT = Q(s,a) + U(s,a)
        where:
        - Q(s,a) = W/N (average value)
        - U(s,a) = c_puct * P(s,a) * sqrt(N) / (1 + n)
        
        Args:
            c_puct: Exploration constant (default 1.41 ≈ sqrt(2))
        
        Returns:
            Best child node according to PUCT formula
        """
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            # Q value: average value of this child
            Q = child.W / child.N if child.N > 0 else 0.0
            
            # U value: exploration bonus
            U = c_puct * child.P * (math.sqrt(self.N) / (1 + child.N))
            
            # PUCT score
            puct_score = Q + U

            if puct_score > best_score:
                best_score = puct_score
                best_child = child
                
        return best_child
    
    def expand(self, network):
        """
        Expand this node by creating child nodes for all legal moves.
        
        Uses neural network to get policy and value estimates.
        
        Args:
            network: Neural network for policy and value prediction
        
        Returns:
            Value estimate from neural network
        """
        # Get policy and value from neural network
        policy_vector, value = network.predict(self.game_state.encode())
        
        # Get legal moves
        legal_moves = self.game_state.legal_moves()
        
        # Get legal policy with heuristic weighting
        legal_policy = get_legal_policy_dict(policy_vector, legal_moves)

        # Create child nodes for each legal move
        for move, prob in legal_policy.items():
            new_state = self.game_state.clone()
            new_state.make_move(move)
            
            child_node = PUCTNode(
                game_state=new_state, 
                parent=self, 
                move=move, 
                prior=prob 
            )
            self.children.append(child_node)
        
        self.is_expanded = True
        return value 
    
    def backpropagate(self, val: int):
        """
        Backpropagate value up the tree.
        
        Updates visit count and value sum for this node and ancestors.
        
        Args:
            val: Value to backpropagate (typically -1, 0, or 1)
        """
        self.N += 1
        self.W += val
        
        if self.parent is not None:
            self.parent.backpropagate(-val)
            # Alternate sign if player changes
            # if self.parent.player != self.player:
            #     self.parent.backpropagate(-val)
            # else:
            #     self.parent.backpropagate(val)


class PUCTSearch:
    """
    PUCT (Polynomial Upper Confidence Tree) search algorithm.
    
    Combines Monte Carlo Tree Search with neural network guidance.
    """
    
    def __init__(self, network, c_puct=1.41):
        """
        Initialize PUCT search.
        
        Args:
            network: Neural network for policy and value guidance
            c_puct: Exploration constant (default 1.41)
        """
        self.network = network
        self.c_puct = c_puct

    def _print_stats(self, root):
        """Print search statistics for debugging."""
        print(f"\n[Search Done] Total Simulations: {root.N}")
        print(f"{'Move':<12} {'N (Visits)':<12} {'Q (Value)':<12} {'P (Prior)':<12}")
        print("-" * 50)
        # Sort children by visit count for display
        for child in sorted(root.children, key=lambda c: c.N, reverse=True):
            move_str = f"{child.move[0]},{child.move[1]}"
            q_val = child.W / child.N if child.N > 0 else 0.0
            print(f"{move_str:<12} {child.N:<12} {q_val:<12.3f} {child.P:<12.3f}")

    def search(self, game_state: Othello, num_simulations: int = 100):
        """
        Perform PUCT search to find the best move.
        
        Args:
            game_state: Current Othello game state
            num_simulations: Number of simulations to run
        
        Returns:
            Best move found by search
        """
        # Create root node
        root = PUCTNode(game_state.clone())
        
        # Expand root node
        root.expand(self.network)

        # Run simulations
        for _ in range(num_simulations):
            node = root
            
            # Selection and expansion phase
            while node.is_expanded and not node.is_terminal():
                next_node = node.best_child(self.c_puct)
                if next_node is None: 
                    break
                node = next_node
            
            # Evaluation phase
            if not node.is_terminal():
                value = node.expand(self.network)
            else:
                # Terminal node: determine winner
                is_over, winner = node.game_state.status()
                if winner == 0: 
                    value = 0
                else:
                    value = 1 if winner == node.game_state.current_player else -1

            # Backpropagation phase
            node.backpropagate(value)
        
        # Print statistics
        self._print_stats(root)
        
        # Return best move (highest visit count)
        best_move = max(root.children, key=lambda c: c.N).move if root.children else None
        return best_move

    def search_for_training(self, game_state: Othello, num_simulations: int = 100):
        root = PUCTNode(game_state.clone())
        root.expand(self.network)

        for _ in range(num_simulations):
            node = root
            # Selection
            while node.is_expanded and not node.is_terminal():
                next_node = node.best_child(self.c_puct)
                if next_node is None: break
                node = next_node
            
            # Expansion & Evaluation
            if not node.is_terminal():
                value = node.expand(self.network)
            else:
                is_over, winner = node.game_state.status()
                if winner == 0: value = 0
                else: value = 1 if winner == node.game_state.current_player else -1

            node.backpropagate(value)

        # تجهيز الـ Policy Vector (64 خانة)
        pi = np.zeros(64)
        for child in root.children:
            r, c = child.move
            pi[r * 8 + c] = child.N
        
        # تحويل الزيارات إلى احتمالات (Normalization)
        if pi.sum() > 0:
            pi /= pi.sum()
        else:
            # في حالة نادرة إذا لم تكن هناك زيارات، وزع الاحتمالات بالتساوي
            pi = np.ones(64) / 64

        # 2. 🔥 الحل الأكيد: إعادة التطبيع لضمان أن المجموع 1.0 بالضبط لـ numpy
        pi = pi / np.sum(pi)
        # في التدريب نختار النقلة بناءً على التوزيع وليس الأفضل دائماً لضمان الاستكشاف [cite: 24]
        move_idx = np.random.choice(len(pi), p=pi)
        best_move = (move_idx // 8, move_idx % 8)
        
        return best_move, pi
    
      

def collect_and_save_data(network, num_games=20, output_file="self_play_data.npz"):
    all_states = []
    all_pis = []
    all_values = []

    for game_idx in range(num_games):
        game = Othello()
        game_history = [] # لتخزين (state, pi, player)

        while not game.status()[0]:
            searcher = PUCTSearch(network)
            # الحصول على النقلة وتوزيع الاحتمالات
            move, pi = searcher.search_for_training(game, num_simulations=200)
            
            # تخزين الحالة (Encoded) والتوزيع واللاعب الحالي
            game_history.append([game.encode(), pi, game.current_player])
            
            game.make_move(move)

        # بعد نهاية المباراة، نحدد القيمة (Value Target) 
        is_over, winner = game.status()
        
        for state, pi, player in game_history:
            all_states.append(state)
            all_pis.append(pi)
            
            # القيمة من منظور اللاعب صاحب الدور (1 فوز، -1 خسارة)
            if winner == 0:
                z = 0
            else:
                z = 1 if winner == player else -1
            all_values.append(z)
        
        print(f"Finished Game {game_idx + 1}")

    # حفظ البيانات كمصفوفات NumPy مضغوطة
    np.savez_compressed(
        output_file,
        states=np.array(all_states),
        pis=np.array(all_pis),
        values=np.array(all_values)
    )
    print(f"Successfully saved {len(all_states)} samples to {output_file}")


def evaluate_agents(model_v1, model_v2, num_games=10):
    """
    Plays two neural network agents against each other using PUCT Search.
    Model V1 will play as Player 1 (Black), and Model V2 as Player 2 (White).
    """
    results = {
        "model_v1_wins": 0,
        "model_v2_wins": 0,
        "draws": 0
    }

    # Initialize PUCT searchers for both models
    searcher1 = PUCTSearch(model_v1)
    searcher2 = PUCTSearch(model_v2)

    for game_id in range(1, num_games + 1):
        print(f"--- Starting Game {game_id} ---")
        game = Othello() # Create a fresh game of Othello
        
        while not game.status()[0]: # While game is not over
            # Determine which searcher to use based on the current player
            current_searcher = searcher1 if game.current_player == 1 else searcher2
            
            # Perform PUCT search to find the best move (e.g., 100 simulations)
            move = current_searcher.search(game, num_simulations=10)
            
            # Execute the move on the board
            game.make_move(move)
        
        # Determine the winner after the game ends
        _, winner = game.status()
        
        if winner == 1:
            print(f"Game {game_id}: Model V1 (Player 1) Wins!")
            results["model_v1_wins"] += 1
        elif winner == -1:
            print(f"Game {game_id}: Model V2 (Player 2) Wins!")
            results["model_v2_wins"] += 1
        else:
            print(f"Game {game_id}: It's a Draw!")
            results["draws"] += 1

    # Print final evaluation summary
    print("\n============================")
    print("FINAL COMPETITION RESULTS")
    print(f"Model V1 Wins: {results['model_v1_wins']}")
    print(f"Model V2 Wins: {results['model_v2_wins']}")
    print(f"Draws: {results['draws']}")
    print("============================\n")
    
    return results


if __name__ == "__main__":
    # Fix for the loading warning: load weights only (ignoring optimizer state)
    current_version = GameNetworkSimple(action_size=64)
    old_version = GameNetworkSimple(action_size=64)

    try:
        # Load the trained model weights
        # current_version.model.load_weights('othello_model_100k_v1.keras', compile=False)
        # print("Model weights loaded successfully! othello_final_pro_model_v1.keras")
        
        current_version.model = tf.keras.models.load_model('othello_model_100k_v1.keras', compile=False)
        print("Model loaded successfully! othello_model_100k_v1.keras")
    except:
        print("Warning: Could not load model weights, AI will be random.")

    try:
        # Load the trained model weights
        # old_version.model.load_weights('othello_final_pro_modelv2.keras')
        # print("Model weights loaded successfully! othello_final_pro_model_v2.keras")
        
        old_version.model = tf.keras.models.load_model('othello_model_100k_v2.keras', compile=False)
        print("Model loaded successfully! othello_model_100k_v2.keras")
    except:
        print("Warning: Could not load model weights, AI will be random.")

    # Run evaluation with PUCT simulations
    stats = evaluate_agents(current_version, old_version, num_games=4)
    print(f"\nFinal Stats: {stats}")