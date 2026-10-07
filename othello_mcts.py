import random
import math
from typing import Optional
from othello import Othello

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
        if self.game_state is None: return True
        """Check if this node represents a terminal game state."""
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
        if not self.children: return None
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            if child.visits == 0:
                ucb1_score = float('inf')
            else:
                exploitation = child.wins / child.visits
                exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                ucb1_score = exploitation + exploration
            if ucb1_score > best_score:
                best_score = ucb1_score
                best_child = child
        return best_child
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
    
    # def simulate(self) -> int:
    #     """
    #     Simulate a random playout from this node to a terminal state.
        
    #     Returns:
    #         The result of the simulation (RED_WIN, YELLOW_WIN, or DRAW)
    #     """
    #     simulation_state = self.game_state.clone()
    #     max_moves = 100 # Safety break
    #     moves_made = 0
    #     while moves_made < max_moves:
    #         is_over, winner = simulation_state.status()
    #         if is_over: return winner
    #         moves = simulation_state.legal_moves()
    #         if not moves:
    #             simulation_state.make_move(None)
    #         else:
    #             simulation_state.make_move(random.choice(moves))
    #         moves_made += 1
    #     _, winner = simulation_state.status()
    #     return winner if winner is not None else 0

    def simulate(self) -> int: #  max
        """
        محاكاة اللعب باستخدام الأوزان لتوجيه الاختيارات (Heuristic Simulation).
        """
        # تعريف الأوزان داخل الدالة أو استدعاؤها من الخارج
       

        simulation_state = self.game_state.clone()
        max_moves = 100 
        moves_made = 0
        
        while moves_made < max_moves:
            is_over, winner = simulation_state.status()
            if is_over: return winner
            
            moves = simulation_state.legal_moves()
            if not moves:
                simulation_state.make_move(None)
            else:
                # --- التعديل هنا ---
                # بدلاً من random.choice(moves)، نختار الحركة ذات الوزن الأعلى
                # أو نستخدم softmax/probabilities بناءً على الأوزان.
                # أبسط طريقة: اختر أفضل حركة بناءً على الجدول
                best_move = max(moves, key=lambda m: WEIGHTS[m[0]][m[1]])
                
                # ملاحظة: لجعل اللعب غير متوقع تماماً، يمكننا إضافة قليل من العشوائية
                # لكن اختيار الحركة الأعلى وزناً سيعطي نتائج قوية جداً
                simulation_state.make_move(best_move)
                
            moves_made += 1
            
        _, winner = simulation_state.status()
        return winner if winner is not None else 0

    def backpropagate(self, result: int):
        """
        Backpropagate the simulation result up the tree.
        
        Args:
            result: The result of the simulation (RED_WIN, YELLOW_WIN, or DRAW)
        """
        self.visits += 1
        # result is the winner (1 for Black, 2 for White, 0 for Draw)
        # We need to update wins from the perspective of the player who made the move to reach this node.
        # This node was reached by a move made by self.parent.game_state.current_player
        if self.parent is not None:
            player_who_moved = self.parent.game_state.current_player
            if result == player_who_moved:
                self.wins += 1
            elif result == 0:
                self.wins += 0.5
            
            self.parent.backpropagate(result)

class MCTS:
    """
    Monte Carlo Tree Search algorithm implementation for Connect Four.
    """
    
    def __init__(self, exploration_weight: float = 1.41):
        """
        Initialize MCTS.
        
        Args:
            exploration_weight: The exploration parameter for UCB1 (default sqrt(2))
        """
        self.exploration_weight = exploration_weight
        
    
    def search(self, game_state: Othello, num_simulations: int = 100) -> int:
        """
        Perform MCTS to find the best move for the current game state.
        
        Args:
            game_state: The current Othello game state
            num_simulations: Number of MCTS simulations to run
        
        Returns:
            The best move (column index) to make
        """
        root = MCTSNode(game_state.clone())
        
        if not root.untried_moves and not root.is_terminal():
            return None 

        for _ in range(num_simulations):
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
            
            # 3. Simulation: Run a random playout from the node
            result = node.simulate()
            
            # 4. Backpropagation: Update statistics up the tree
            node.backpropagate(result)
        
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
        
        Args:
            game_state: The current Connect Four game state
            num_simulations: Number of MCTS simulations to run
        
        Returns:
            Dictionary mapping moves to their statistics
        """
        root = MCTSNode(game_state.clone())
        
        if not root.untried_moves and not root.is_terminal():
            return None 

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
            result = node.simulate()
            
            # Backpropagation
            node.backpropagate(result)
        
        # Collect statistics for each move
        statistics = {}
        for child in root.children:
            win_rate = child.wins / child.visits if child.visits > 0 else 0
            statistics[child.move] = {
                'visits': child.visits,
                'wins': child.wins,
                'win_rate': win_rate
            }
        
        return statistics

def play_mcts_vs_mcts(sims_black: int = 1000, sims_white: int = 1000):
    """
    Mode where two MCTS agents play against each other.
    """
    game = Othello()
    mcts = MCTS()
    print(f"Othello: MCTS Black ({sims_black} sims) vs MCTS White ({sims_white} sims)")
    
    move_count = 1
    while True:
        game.display()
        is_over, winner = game.status()
        
        if is_over:
            print("\n--- GAME OVER ---")
            black_score = sum(row.count(game.BLACK) for row in game.board)
            white_score = sum(row.count(game.WHITE) for row in game.board)
            print(f"Final Score - Black: {black_score}, White: {white_score}")
            if winner == Othello.BLACK: print("Winner: BLACK (AI)")
            elif winner == Othello.WHITE: print("Winner: WHITE (AI)")
            else: print("It's a DRAW!")
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

def play_human_vs_mcts(human_player: int = Othello.WHITE, 
                       num_simulations: int = 100):
    """
    Main loop for playing Human vs MCTS AI in the terminal.
    """
    game = Othello()
    mcts = MCTS()
    print("Othello: Human (Black) vs MCTS AI (White)")
    
    while True:
        game.display()
        is_over, winner = game.status()
        
        if is_over:
            print("\n--- GAME OVER ---")
            if winner == Othello.BLACK: print("Winner: BLACK (Human)")
            elif winner == Othello.WHITE: print("Winner: WHITE (AI)")
            else: print("It's a DRAW!")
            break
            
        moves = game.legal_moves()
        if not moves:
            print(f"No moves for {('Black' if game.current_player == Othello.BLACK else 'White')}. Passing.")
            game.make_move(None)
            continue
            
        if game.current_player == Othello.BLACK:
            print(f"Legal moves: {moves}")
            try:
                move_str = input("Enter move (row,col): ")
                r, c = map(int, move_str.split(','))
                if (r, c) in moves:
                    game.make_move((r, c))
                else:
                    print("Invalid move!")
            except:
                print("Format error! Use: row,col")
        else:
            print("AI is thinking...")
            best_move = mcts.search(game, num_simulations=100)
            print(f"AI plays: {best_move}")
            game.make_move(best_move)
