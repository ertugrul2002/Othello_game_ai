import random
import math
import numpy as np
from typing import Optional
from othello import Othello

# Position weights for Othello board evaluation
# Corners (100) are most valuable, edges near corners (-20, -50) are dangerous
WEIGHTS = [
    [100, -20,  10,   5,   5,  10, -20, 100],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [ 10,  -2,   5,   1,   1,   5,  -2,  10],
    [  5,  -2,   1,   0,   0,   1,  -2,   5],
    [  5,  -2,   1,   0,   0,   1,  -2,   5],
    [ 10,  -2,   5,   1,   1,   5,  -2,  10],
    [-20, -50,  -2,  -2,  -2,  -2, -50, -20],
    [100, -20,  10,   5,   5,  10, -20, 100]
]


class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search tree.
    Each node corresponds to a game state.
    """
    
    def __init__(self, game_state: Othello, parent: Optional['MCTSNode'] = None, 
                 move: Optional[int] = None):
        """
        Initialize a new MCTS node.
        
        Args:
            game_state: The Othello game state at this node
            parent: The parent node (None for root)
            move: The move that led to this state from parent
        """
        self.game_state = game_state
        self.parent = parent
        self.move = move  # The move that led to this state
        
        self.children = []  # List of child nodes
        self.untried_moves = game_state.legal_moves()  # Moves not yet explored
        
        self.wins = 0  # Number of wins from this node
        self.visits = 0  # Number of times this node has been visited
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible moves from this node have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Check if this node represents a terminal game state."""
        if self.game_state is None: 
            return True
        is_over, _ = self.game_state.status()
        return is_over
    
    def best_child(self, exploration_weight: float = 1.41) -> 'MCTSNode':
        """
        Select the best child using the UCB1 (Upper Confidence Bound) formula.
        
        UCB1 = (wins / visits) + exploration_weight * sqrt(ln(parent_visits) / visits)
        
        Args:
            exploration_weight: The exploration parameter (default sqrt(2))
        
        Returns:
            The child node with the highest UCB1 value
        """
        if not self.children: 
            return None
            
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                # Unvisited nodes get infinite priority
                ucb1_score = float('inf')
            else:
                # Exploitation term: win rate
                exploitation = child.wins / child.visits
                # Exploration term: encourages trying less-visited nodes
                exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                ucb1_score = exploitation + exploration
                
            if ucb1_score > best_score:
                best_score = ucb1_score
                best_child = child
                
        return best_child
    
    def expand(self) -> 'MCTSNode':
        """
        Expand the tree by creating a new child node for an untried move.
        
        Returns:
            The newly created child node
        """
        # Select a random untried move
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)
        
        # Create a new game state by applying the move
        new_state = self.game_state.clone()
        new_state.make_move(move)
        
        # Create a new child node
        child_node = MCTSNode(new_state, parent=self, move=move)
        self.children.append(child_node)
        
        return child_node
    
    def simulate(self, temperature: float = 1.0) -> int:
        """
        Simulate a playout from this node using probabilistic heuristic guidance.
        Uses softmax over position weights to balance exploration and exploitation.
        
        Args:
            temperature: Controls randomness (higher = more random, lower = more greedy)
                        temperature=1.0 is standard softmax
                        temperature=0.5 is more greedy
                        temperature=2.0 is more exploratory
        
        Returns:
            The result of the simulation (1 for Black, 2 for White, 0 for Draw)
        """
        simulation_state = self.game_state.clone()
        max_moves = 100  # Safety limit to prevent infinite loops
        moves_made = 0
        
        while moves_made < max_moves:
            is_over, winner = simulation_state.status()
            if is_over: 
                return winner
            
            moves = simulation_state.legal_moves()
            if not moves:
                # No legal moves, pass turn
                simulation_state.make_move(None)
            else:
                # Use softmax to convert weights to probabilities
                move_weights = np.array([WEIGHTS[r][c] for r, c in moves], dtype=np.float64)
                
                # Apply temperature scaling
                move_weights = move_weights / temperature
                
                # Softmax: convert weights to probabilities
                # Subtract max for numerical stability
                exp_weights = np.exp(move_weights - np.max(move_weights))
                move_probabilities = exp_weights / np.sum(exp_weights)
                
                # Sample a move based on probabilities
                selected_index = np.random.choice(len(moves), p=move_probabilities)
                selected_move = moves[selected_index]
                
                simulation_state.make_move(selected_move)
                
            moves_made += 1
            
        # If max moves reached, return current winner
        _, winner = simulation_state.status()
        return winner if winner is not None else 0

    def backpropagate(self, result: int):
        """
        Backpropagate the simulation result up the tree.
        Updates visit counts and win statistics for all ancestor nodes.
        
        Args:
            result: The result of the simulation (1 for Black, 2 for White, 0 for Draw)
        """
        self.visits += 1
        
        # Update wins from the perspective of the player who made the move to reach this node
        if self.parent is not None:
            player_who_moved = self.parent.game_state.current_player
            
            if result == player_who_moved:
                # Win for the player who moved
                self.wins += 1
            elif result == 0:
                # Draw gives half a win
                self.wins += 0.5
            # Loss gives 0 (no change needed)
            
            # Recursively backpropagate to parent
            self.parent.backpropagate(result)


class MCTS:
    """
    Improved Monte Carlo Tree Search algorithm implementation for Othello.
    
    Key improvements:
    1. Probabilistic heuristic simulation using softmax
    2. Configurable exploration weight
    3. Optional Dirichlet noise for root exploration
    4. Temperature-based move selection
    """
    
    def __init__(self, exploration_weight: float = 1.41, simulation_temperature: float = 1.0):
        """
        Initialize MCTS with configurable parameters.
        
        Args:
            exploration_weight: The exploration parameter for UCB1 (default sqrt(2))
                               Higher values encourage more exploration
                               Typical range: 0.5 to 2.0
            simulation_temperature: Temperature for softmax in simulation
                                   Higher = more random, lower = more greedy
        """
        self.exploration_weight = exploration_weight
        self.simulation_temperature = simulation_temperature
    
    def search(self, game_state: Othello, num_simulations: int = 1000, 
               use_dirichlet: bool = False, dirichlet_alpha: float = 0.3,
               dirichlet_epsilon: float = 0.25) -> int:
        """
        Perform MCTS to find the best move for the current game state.
        
        Args:
            game_state: The current Othello game state
            num_simulations: Number of MCTS simulations to run
            use_dirichlet: Whether to add Dirichlet noise to root for exploration
            dirichlet_alpha: Alpha parameter for Dirichlet distribution
            dirichlet_epsilon: Weight of noise vs original probabilities
        
        Returns:
            The best move to make
        """
        root = MCTSNode(game_state.clone())
        
        # Handle case where no moves are available
        if not root.untried_moves and not root.is_terminal():
            return None
        
        # Run simulations
        for simulation_num in range(num_simulations):
            node = root
            
            # 1. Selection: Traverse the tree using UCB1 until we reach a leaf
            while not node.is_terminal() and node.is_fully_expanded():
                next_node = node.best_child(self.exploration_weight)
                if next_node is None:
                    break
                node = next_node
            
            # 2. Expansion: If the node is not terminal, expand it
            if not node.is_terminal() and not node.is_fully_expanded():
                expanded_node = node.expand()
                if expanded_node: 
                    node = expanded_node
            
            # 3. Simulation: Run a probabilistic heuristic playout from the node
            result = node.simulate(temperature=self.simulation_temperature)
            
            # 4. Backpropagation: Update statistics up the tree
            node.backpropagate(result)
        
        # Optional: Add Dirichlet noise to root children for exploration
        # This is useful for generating diverse training data
        if use_dirichlet and root.children:
            noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
            for i, child in enumerate(root.children):
                # Mix noise with visit-based probability
                child.visits = int((1 - dirichlet_epsilon) * child.visits + 
                                  dirichlet_epsilon * noise[i] * num_simulations)
        
        # Return the move with the highest visit count (most robust choice)
        best_move = None
        best_visits = -1
        
        for child in root.children:
            if child.visits > best_visits:
                best_visits = child.visits
                best_move = child.move
        
        return best_move
    
    def get_move_statistics(self, game_state: Othello, 
                           num_simulations: int = 1000) -> dict:
        """
        Get detailed statistics for all possible moves.
        Useful for analyzing MCTS behavior and generating training data.
        
        Args:
            game_state: The current Othello game state
            num_simulations: Number of MCTS simulations to run
        
        Returns:
            Dictionary mapping moves to their statistics (visits, wins, win_rate)
        """
        root = MCTSNode(game_state.clone())
        
        if not root.untried_moves and not root.is_terminal():
            return None
        
        # Run simulations
        for _ in range(num_simulations):
            node = root
            
            # Selection
            while not node.is_terminal() and node.is_fully_expanded():
                next_node = node.best_child(self.exploration_weight)
                if next_node is None:
                    break
                node = next_node
            
            # Expansion
            if not node.is_terminal() and not node.is_fully_expanded():
                expanded_node = node.expand()
                if expanded_node: 
                    node = expanded_node
            
            # Simulation
            result = node.simulate(temperature=self.simulation_temperature)
            
            # Backpropagation
            node.backpropagate(result)
        
        # Collect statistics for each move
        statistics = {}
        total_visits = sum(child.visits for child in root.children)
        
        for child in root.children:
            win_rate = child.wins / child.visits if child.visits > 0 else 0
            visit_probability = child.visits / total_visits if total_visits > 0 else 0
            
            statistics[child.move] = {
                'visits': child.visits,
                'wins': child.wins,
                'win_rate': win_rate,
                'visit_probability': visit_probability
            }
        
        return statistics


def play_mcts_vs_mcts(sims_black: int = 1000, sims_white: int = 1000,
                      exploration_weight: float = 1.41, 
                      simulation_temperature: float = 1.0):
    """
    Mode where two MCTS agents play against each other.
    
    Args:
        sims_black: Number of simulations for Black player
        sims_white: Number of simulations for White player
        exploration_weight: UCB1 exploration parameter
        simulation_temperature: Temperature for simulation softmax
    """
    game = Othello()
    mcts = MCTS(exploration_weight=exploration_weight, 
                simulation_temperature=simulation_temperature)
    
    print(f"Othello: MCTS Black ({sims_black} sims) vs MCTS White ({sims_white} sims)")
    print(f"Exploration weight: {exploration_weight}, Simulation temperature: {simulation_temperature}")
    
    move_count = 1
    while True:
        game.display()
        is_over, winner = game.status()
        
        if is_over:
            print("\n--- GAME OVER ---")
            black_score = sum(row.count(game.BLACK) for row in game.board)
            white_score = sum(row.count(game.WHITE) for row in game.board)
            print(f"Final Score - Black: {black_score}, White: {white_score}")
            if winner == Othello.BLACK: 
                print("Winner: BLACK (AI)")
            elif winner == Othello.WHITE: 
                print("Winner: WHITE (AI)")
            else: 
                print("It's a DRAW!")
            break
            
        moves = game.legal_moves()
        if not moves:
            print(f"No moves for {'Black' if game.current_player == Othello.BLACK else 'White'}. Passing.")
            game.make_move(None)
            continue
            
        current_sims = sims_black if game.current_player == Othello.BLACK else sims_white
        player_name = "Black" if game.current_player == Othello.BLACK else "White"
        
        print(f"Move {move_count}: AI {player_name} is thinking with {current_sims} simulations...")
        best_move = mcts.search(game, num_simulations=current_sims)
        print(f"AI {player_name} plays: {best_move}")
        game.make_move(best_move)
        move_count += 1
        print("-" * 30)


def play_human_vs_mcts(human_player: int = Othello.BLACK, 
                       num_simulations: int = 1000,
                       exploration_weight: float = 1.41,
                       simulation_temperature: float = 1.0):
    """
    Main loop for playing Human vs MCTS AI in the terminal.
    
    Args:
        human_player: Which player is human (Othello.BLACK or Othello.WHITE)
        num_simulations: Number of simulations for AI
        exploration_weight: UCB1 exploration parameter
        simulation_temperature: Temperature for simulation softmax
    """
    game = Othello()
    mcts = MCTS(exploration_weight=exploration_weight,
                simulation_temperature=simulation_temperature)
    
    human_color = "Black" if human_player == Othello.BLACK else "White"
    ai_color = "White" if human_player == Othello.BLACK else "Black"
    print(f"Othello: Human ({human_color}) vs MCTS AI ({ai_color})")
    print(f"AI using {num_simulations} simulations per move")
    
    while True:
        game.display()
        is_over, winner = game.status()
        
        if is_over:
            print("\n--- GAME OVER ---")
            black_score = sum(row.count(game.BLACK) for row in game.board)
            white_score = sum(row.count(game.WHITE) for row in game.board)
            print(f"Final Score - Black: {black_score}, White: {white_score}")
            
            if winner == human_player: 
                print(f"Winner: {human_color} (Human) - Congratulations!")
            elif winner != 0: 
                print(f"Winner: {ai_color} (AI) - Better luck next time!")
            else: 
                print("It's a DRAW!")
            break
            
        moves = game.legal_moves()
        if not moves:
            print(f"No moves for {'Black' if game.current_player == Othello.BLACK else 'White'}. Passing.")
            game.make_move(None)
            continue
            
        if game.current_player == human_player:
            # Human turn
            print(f"Legal moves: {moves}")
            try:
                move_str = input("Enter move (row,col): ")
                r, c = map(int, move_str.split(','))
                if (r, c) in moves:
                    game.make_move((r, c))
                else:
                    print("Invalid move! Try again.")
            except:
                print("Format error! Use: row,col (e.g., 2,3)")
        else:
            # AI turn
            print(f"AI {ai_color} is thinking...")
            best_move = mcts.search(game, num_simulations=num_simulations)
            print(f"AI {ai_color} plays: {best_move}")
            game.make_move(best_move)


if __name__ == "__main__":
    # Example usage: MCTS vs MCTS with improved settings
    # Try different configurations to find what works best
    
    # Configuration 1: Balanced (recommended for strong play)
    # play_mcts_vs_mcts(sims_black=1000, sims_white=1000, 
    #                   exploration_weight=1.41, simulation_temperature=1.0)
    
    # Configuration 2: More exploratory (good for generating diverse training data)
    # play_mcts_vs_mcts(sims_black=1000, sims_white=1000,
    #                   exploration_weight=1.8, simulation_temperature=1.2)
    
    # Configuration 3: More greedy (stronger but less diverse)
    play_mcts_vs_mcts(sims_black=800, sims_white=100,
                      exploration_weight=1.2, simulation_temperature=0.8)
    
    # Human vs AI
    # play_human_vs_mcts(human_player=Othello.BLACK, num_simulations=1000,
    #                    exploration_weight=1.41, simulation_temperature=1.0)
