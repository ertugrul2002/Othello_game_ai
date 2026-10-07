import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw
import time
import threading
from othello import Othello
import threading
from othello_network import GameNetworkSimple
from othello_puct import PUCTSearch
from othello_mcts_improved import MCTS
import tensorflow as tf
import os

class OthelloGUI:
    def __init__(self, root):
        # Initialize the main window and game logic
        self.root = root
        self.root.title("Othello Ultimate with MCTS/PUCT")
        self.root.configure(bg="#1e1e1e")
        self.game = Othello()
        
        # Initialize AI algorithm choice (will be set by popup dialog)
        self.ai_algorithm = None
        
        # Initialize AI network
        self.ai_net = GameNetworkSimple(action_size=64)
        try:
            # Load the trained model weights
            # self.ai_net.model.load_weights('my_othello_modle_1000.h5',by_name=False)
            # print("Model weights loaded successfully! my_othello_modle_1000.h5")
            model_path = 'my_othello_modle_1000_v1.h5'
            if os.path.exists(model_path):
                self.ai_net.model.load_weights(model_path)
                print("Model weights loaded successfully!")
            else:
                print(f"Error: The file {model_path} was not found in {os.getcwd()}")
            
            # self.ai_net.model = tf.keras.models.load_model('othello_model_100k_v1.keras', compile=False)
            # print("Model loaded successfully! othello_model_100k_v1.keras")
        except:
            print("Warning: Could not load model weights, AI will be random.")
        
        self.BG_COLOR = "#1e1e1e"
        self.SIDEBAR_COLOR = "#2d2d2d"
        self.BOARD_COLOR = "#235d3a"
        self.GRID_COLOR = "#1a432a"
        self.TEXT_FRAME_BG = "#121212"
        
        self.ai_running = False
        self.mcts_vs_mcts_active = False
        self.mcts_vs_human_active = False
        self.human_player = Othello.BLACK
        
        # Store top 3 moves for display
        self.top_moves = []
        
        self.load_assets()
        self.setup_ui()
        self.draw_board()
        self.animating = False
        
        # Show algorithm selection dialog
        self.show_algorithm_selection()

    def show_algorithm_selection(self):
        """Show a dialog to select between MCTS and PUCT before starting the game"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select AI Algorithm")
        dialog.geometry("450x280")
        dialog.configure(bg=self.SIDEBAR_COLOR)
        dialog.resizable(False, False)
        
        # Center the dialog on the main window
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Get screen dimensions to center dialog
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - 225
        y = (self.root.winfo_screenheight() // 2) - 140
        dialog.geometry(f"+{x}+{y}")
        
        # Title
        title_label = tk.Label(dialog, text="Choose AI Algorithm", font=("Segoe UI", 18, "bold"), 
                bg=self.SIDEBAR_COLOR, fg="#2ecc71")
        title_label.pack(pady=(40, 10))
        
        # Description
        desc_label = tk.Label(dialog, text="Select which algorithm to use for AI moves", font=("Segoe UI", 11), 
                bg=self.SIDEBAR_COLOR, fg="#aaaaaa")
        desc_label.pack(pady=(0, 40))
        
        def select_mcts():
            self.ai_algorithm = "MCTS"
            dialog.destroy()
        
        def select_puct():
            self.ai_algorithm = "PUCT"
            dialog.destroy()
        
        # Button container
        button_frame = tk.Frame(dialog, bg=self.SIDEBAR_COLOR)
        button_frame.pack(pady=20)
        
        # MCTS Button
        mcts_btn = tk.Button(button_frame, text="MCTS", command=select_mcts, bg="#3498db", 
                           font=("Segoe UI", 13, "bold"), fg="white", pady=15, 
                           relief="flat", cursor="hand2", width=14)
        mcts_btn.pack(side="left", padx=15)
        
        # PUCT Button
        puct_btn = tk.Button(button_frame, text="PUCT", command=select_puct, bg="#9b59b6", 
                           font=("Segoe UI", 13, "bold"), fg="white", pady=15, 
                           relief="flat", cursor="hand2", width=14)
        puct_btn.pack(side="left", padx=15)

    def load_assets(self):
        """Load and resize piece images"""
        try:
            # Create simple colored circles if images are not found
            self.black_pil = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
            self.white_pil = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
            
            draw_b = ImageDraw.Draw(self.black_pil)
            draw_b.ellipse((2, 2, 48, 48), fill="black", outline="gray")
            
            draw_w = ImageDraw.Draw(self.white_pil)
            draw_w.ellipse((2, 2, 48, 48), fill="white", outline="gray")
            
            self.black_photo = ImageTk.PhotoImage(self.black_pil)
            self.white_photo = ImageTk.PhotoImage(self.white_pil)
        except Exception as e:
            print(f"Error creating assets: {e}")

    def setup_ui(self):
        """Setup the layout with sidebar on the left and board on the right"""
        self.main_container = tk.Frame(self.root, bg=self.BG_COLOR)
        self.main_container.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Sidebar (Left)
        self.sidebar = tk.Frame(self.main_container, bg=self.SIDEBAR_COLOR, width=220, padx=15, pady=20)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        
        tk.Label(self.sidebar, text="CONTROL PANEL", font=("Segoe UI", 14, "bold"), 
                bg=self.SIDEBAR_COLOR, fg="white").pack(pady=(0, 20), fill="x")
        
        btn_style = {"font": ("Segoe UI", 10, "bold"), "fg": "white", "pady": 10, 
                    "relief": "flat", "cursor": "hand2"}
        tk.Button(self.sidebar, text="NEW GAME", command=self.reset_game, bg="#2ecc71", **btn_style).pack(fill="x", pady=5)
        tk.Button(self.sidebar, text="UNDO MOVE", command=self.undo_move, bg="#e74c3c", **btn_style).pack(fill="x", pady=5)
        
        self.btn_play_black = tk.Button(self.sidebar, text="PLAY AS BLACK", 
                                       command=lambda: self.start_human_vs_mcts(Othello.BLACK), 
                                       bg="#3498db", **btn_style)
        self.btn_play_black.pack(fill="x", pady=5)

        self.btn_play_white = tk.Button(self.sidebar, text="PLAY AS WHITE", 
                                       command=lambda: self.start_human_vs_mcts(Othello.WHITE), 
                                       bg="#9b59b6", **btn_style)
        self.btn_play_white.pack(fill="x", pady=5)
        
        self.btn_mcts_mcts = tk.Button(self.sidebar, text="AI vs AI", command=self.start_mcts_vs_mcts, 
                                      bg="#95a5a6", **btn_style)
        self.btn_mcts_mcts.pack(fill="x", pady=5)
        
        # Analysis button
        self.btn_analysis = tk.Button(self.sidebar, text="ANALYZE", command=self.analyze_position, 
                                     bg="#f39c12", **btn_style)
        self.btn_analysis.pack(fill="x", pady=5)
        
        tk.Frame(self.sidebar, bg=self.SIDEBAR_COLOR).pack(expand=True, fill="both")
        tk.Button(self.sidebar, text="EXIT", command=self.root.quit, bg="#34495e", **btn_style).pack(fill="x", pady=(10, 0))
        
        # Board Area (Right)
        self.board_area = tk.Frame(self.main_container, bg=self.BG_COLOR)
        self.board_area.pack(side="right", padx=20)
        
        # Current Turn Frame (Top)
        self.turn_frame = tk.Frame(self.board_area, bg=self.SIDEBAR_COLOR, padx=20, pady=10, 
                                  highlightbackground="#333333", highlightthickness=1)
        self.turn_frame.pack(fill="x", pady=(0, 15))
        
        self.turn_title = tk.Label(self.turn_frame, text="CURRENT TURN: ", 
                                  font=("Segoe UI", 12, "bold"), bg=self.SIDEBAR_COLOR, fg="#2ecc71")
        self.turn_title.pack(side="left")
        
        self.turn_value = tk.Label(self.turn_frame, text="BLACK", font=("Segoe UI", 12, "bold"), 
                                  bg=self.SIDEBAR_COLOR)
        self.turn_value.pack(side="left")
        
        # Algorithm display
        self.algo_label = tk.Label(self.turn_frame, text="", font=("Segoe UI", 10), 
                                  bg=self.SIDEBAR_COLOR, fg="#2ecc71")
        self.algo_label.pack(side="right")
        
        # Container for board and analysis panel
        self.board_container = tk.Frame(self.board_area, bg=self.BG_COLOR)
        self.board_container.pack()
        
        # Left side: row numbers + board
        left_side = tk.Frame(self.board_container, bg=self.BG_COLOR)
        left_side.pack(side="left")
        
        # Row numbers (0-7)
        row_numbers_frame = tk.Frame(left_side, bg=self.BG_COLOR, width=40)
        row_numbers_frame.pack(side="left", padx=(0, 5))
        
        # Empty space for alignment with column numbers
        tk.Label(row_numbers_frame, text="", bg=self.BG_COLOR, width=5).pack()
        
        for i in range(8):
            tk.Label(row_numbers_frame, text=str(i), font=("Segoe UI", 12, "bold"), 
                    bg=self.BG_COLOR, fg="#2ecc71", width=3, height=3).pack()
        
        # Board with column numbers
        board_and_cols = tk.Frame(left_side, bg=self.BG_COLOR)
        board_and_cols.pack(side="left")
        
        # Column numbers (0-7)
        col_numbers_frame = tk.Frame(board_and_cols, bg=self.BG_COLOR)
        col_numbers_frame.pack()
        
        for i in range(8):
            tk.Label(col_numbers_frame, text=str(i), font=("Segoe UI", 12, "bold"), 
                    bg=self.BG_COLOR, fg="#2ecc71", width=6).pack(side="left")
        
        # The Canvas
        self.canvas = tk.Canvas(board_and_cols, width=480, height=480, bg=self.BOARD_COLOR, 
                               highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Analysis Panel (Right of board)
        self.analysis_panel = tk.Frame(self.board_container, bg=self.SIDEBAR_COLOR, width=220, 
                                      padx=10, pady=10, highlightbackground="#333333", highlightthickness=1)
        self.analysis_panel.pack(side="left", padx=(15, 0), fill="both")
        self.analysis_panel.pack_propagate(False)
        
        tk.Label(self.analysis_panel, text="TOP 3 MOVES", font=("Segoe UI", 11, "bold"), 
                bg=self.SIDEBAR_COLOR, fg="#2ecc71").pack(pady=(0, 10), fill="x")
        
        # Scrollable frame for moves
        self.moves_frame = tk.Frame(self.analysis_panel, bg=self.SIDEBAR_COLOR)
        self.moves_frame.pack(fill="both", expand=True)
        
        # Score Frame (Bottom)
        self.score_container = tk.Frame(self.board_area, bg=self.SIDEBAR_COLOR, padx=20, pady=10, 
                                       highlightbackground="#545353", highlightthickness=1)
        self.score_container.pack(fill="x", pady=(15, 0))

        tk.Label(self.score_container, text="SCORE: ", font=("Segoe UI", 11, "bold"), 
                bg=self.SIDEBAR_COLOR, fg="#2ecc71").pack(side="left")

        self.black_score_label = tk.Label(self.score_container, text="BLACK: 2", 
                                         font=("Segoe UI", 11, "bold"), bg=self.SIDEBAR_COLOR, fg="#aaaaaa")
        self.black_score_label.pack(side="left", padx=(20, 40))
        
        self.white_score_label = tk.Label(self.score_container, text="WHITE: 2", 
                                         font=("Segoe UI", 11, "bold"), bg=self.SIDEBAR_COLOR, fg="white")
        self.white_score_label.pack(side="left")

    def animate_flip(self, r, c, to_player):
        """Animate piece flipping"""
        steps = 6
        target_pil = self.black_pil if to_player == Othello.BLACK else self.white_pil
        source_pil = self.white_pil if to_player == Othello.BLACK else self.black_pil
        for i in range(steps, 0, -1):
            w = max(1, int(50 * (i / steps)))
            img = source_pil.resize((w, 50), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.canvas.delete(f"piece_{r}_{c}")
            self.canvas.create_image(c*60+30, r*60+30, image=photo, tags=f"piece_{r}_{c}")
            self.canvas.image_ref = photo
            self.root.update()
            time.sleep(0.01)
        for i in range(1, steps + 1):
            w = int(50 * (i / steps))
            img = target_pil.resize((w, 50), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.canvas.delete(f"piece_{r}_{c}")
            self.canvas.create_image(c*60+30, r*60+30, image=photo, tags=f"piece_{r}_{c}")
            self.canvas.image_ref = photo
            self.root.update()
            time.sleep(0.01)

    def draw_board(self):
        """Draw the board with grid"""
        self.canvas.delete("all")
        
        # Draw grid lines
        for i in range(9):
            self.canvas.create_line(i*60, 0, i*60, 480, fill=self.GRID_COLOR)
            self.canvas.create_line(0, i*60, 480, i*60, fill=self.GRID_COLOR)
        
        # Draw legal move indicators
        legal_moves = self.game.legal_moves()
        for r, c in legal_moves:
            self.canvas.create_oval(c*60+25, r*60+25, c*60+35, r*60+35, fill="white", outline="")
            
        # Draw pieces
        for r in range(8):
            for c in range(8):
                piece = self.game.board[r][c]
                if piece == Othello.BLACK:
                    self.canvas.create_image(c*60+30, r*60+30, image=self.black_photo, tags=f"piece_{r}_{c}")
                elif piece == Othello.WHITE:
                    self.canvas.create_image(c*60+30, r*60+30, image=self.white_photo, tags=f"piece_{r}_{c}")
        
        self.update_info()

    def update_info(self):
        """Update game information display"""
        if self.game.current_player == Othello.BLACK:
            self.turn_value.config(text="BLACK", fg="#aaaaaa")
        else:
            self.turn_value.config(text="WHITE", fg="white")
        
        # Update algorithm display
        if self.ai_algorithm:
            self.algo_label.config(text=f"Algorithm: {self.ai_algorithm}")
            
        black_score = sum(row.count(Othello.BLACK) for row in self.game.board)
        white_score = sum(row.count(Othello.WHITE) for row in self.game.board)
        
        self.black_score_label.config(text=f"BLACK: {black_score}")
        self.white_score_label.config(text=f"WHITE: {white_score}")
        
        is_over, winner = self.game.status()
        if is_over:
            self.mcts_vs_mcts_active = False
            self.mcts_vs_human_active = False
            msg = "Game Over! "
            if winner == Othello.BLACK: msg += "Black Wins!"
            elif winner == Othello.WHITE: msg += "White Wins!"
            else: msg += "It's a Draw!"
            messagebox.showinfo("Game Over", msg)

    def update_analysis_panel(self):
        """Update the analysis panel with top 3 moves"""
        # Clear previous moves
        for widget in self.moves_frame.winfo_children():
            widget.destroy()
        
        # Display top 3 moves
        for i, move_info in enumerate(self.top_moves[:3]):
            move, n, q, p, win_rate = move_info
            r, c = move
            
            # Create a frame for each move
            move_frame = tk.Frame(self.moves_frame, bg=self.TEXT_FRAME_BG, 
                                 highlightbackground="#444444", highlightthickness=1)
            move_frame.pack(fill="x", pady=5)
            
            # Move position
            tk.Label(move_frame, text=f"Move: ({r},{c})", font=("Segoe UI", 9, "bold"), 
                    bg=self.TEXT_FRAME_BG, fg="#2ecc71").pack(anchor="w", padx=5, pady=(5, 0))
            
            # Visits
            tk.Label(move_frame, text=f"Visits: {n}", font=("Segoe UI", 8), 
                    bg=self.TEXT_FRAME_BG, fg="#aaaaaa").pack(anchor="w", padx=5)
            
            # Q Value
            tk.Label(move_frame, text=f"Q Value: {q:.3f}", font=("Segoe UI", 8), 
                    bg=self.TEXT_FRAME_BG, fg="#aaaaaa").pack(anchor="w", padx=5)
            
            # Win Rate
            tk.Label(move_frame, text=f"Win Rate: {win_rate:.1%}", font=("Segoe UI", 8), 
                    bg=self.TEXT_FRAME_BG, fg="#aaaaaa").pack(anchor="w", padx=5)
            
            # Policy probability
            tk.Label(move_frame, text=f"Policy: {p:.3f}", font=("Segoe UI", 8), 
                    bg=self.TEXT_FRAME_BG, fg="#aaaaaa").pack(anchor="w", padx=5, pady=(0, 5))

    def make_move_with_animation(self, move):
        """Make a move with animation"""
        if self.animating: return
        self.animating = True
        old_board = [row[:] for row in self.game.board]
        current_player = self.game.current_player
        
        self.game.make_move(move)
        
        if move:
            r, c = move
            photo = self.black_photo if current_player == Othello.BLACK else self.white_photo
            self.canvas.create_image(c*60+30, r*60+30, image=photo, tags=f"piece_{r}_{c}")
            self.root.update()
            
            new_board = self.game.board
            for ri in range(8):
                for ci in range(8):
                    if old_board[ri][ci] != Othello.EMPTY and old_board[ri][ci] != new_board[ri][ci]:
                        self.animate_flip(ri, ci, new_board[ri][ci])
        
        self.draw_board()
        self.animating = False
        
        is_over, _ = self.game.status()
        if is_over:
            self.update_info()
            return

        legal_moves = self.game.legal_moves()
        
        if not legal_moves:
            # If no legal moves, pass the turn
            player_name = "Black" if self.game.current_player == Othello.BLACK else "White"
            messagebox.showinfo("Pass", f"{player_name} has no legal moves. Passing turn...")
            self.root.after(500, lambda: self.make_move_with_animation(None))
            return

        # Trigger AI move if needed
        if self.mcts_vs_human_active and self.game.current_player != self.human_player:
            self.root.after(500, self.ai_move)
        elif self.mcts_vs_mcts_active:
            self.root.after(500, self.ai_move)

    def on_click(self, event):
        """Handle board click"""
        if self.animating or self.ai_running: return
        if self.mcts_vs_mcts_active: return
        if self.mcts_vs_human_active and self.game.current_player != self.human_player: return
        
        c, r = event.x // 60, event.y // 60
        
        if 0 <= r < 8 and 0 <= c < 8:
            legal_moves = self.game.legal_moves()
            if (r, c) in legal_moves:
                self.make_move_with_animation((r, c))
                
                # If MCTS vs Human is active and it's AI's turn
                if self.mcts_vs_human_active and self.game.current_player != self.human_player:
                    self.root.after(500, self.ai_move)

    def ai_move(self):
        """Execute AI move"""
        if self.mcts_vs_human_active and self.game.current_player == self.human_player:
            return
        
        if self.ai_running or self.game.status()[0]: return
        if not self.ai_algorithm:
            messagebox.showerror("Error", "AI algorithm not selected!")
            return
        
        self.ai_running = True
        
        def run_ai():
            try:
                if self.ai_algorithm == "PUCT":
                    # Use PUCT search
                    searcher = PUCTSearch(self.ai_net)
                    move, root_node = self.search_puct_with_stats(searcher, self.game, num_simulations=200)
                elif self.ai_algorithm == "MCTS":
                    # Use MCTS search
                    mcts = MCTS()
                    move, root_node = self.search_mcts_with_stats(mcts, self.game, num_simulations=400)
                else:
                    move = None
                    root_node = None
                
                self.root.after(0, lambda: self.complete_ai_move(move, root_node))
            except Exception as e:
                print(f"Error during AI move: {e}")
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.complete_ai_move(None, None))
            
        threading.Thread(target=run_ai).start()

    def search_puct_with_stats(self, searcher, game, num_simulations):
        """Run PUCT search and return move + root node for statistics"""
        from othello_puct import PUCTNode
        root = PUCTNode(game.clone())
        root.expand(self.ai_net)
        
        # Run simulations
        for _ in range(num_simulations):
            node = root
            
            # Selection and expansion
            while node.is_expanded and not node.is_terminal():
                next_node = node.best_child(searcher.c_puct)
                if next_node is None: break
                node = next_node
            
            # Evaluation
            if not node.is_terminal():
                value = node.expand(self.ai_net)
            else:
                is_over, winner = node.game_state.status()
                if winner == 0:
                    value = 0
                else:
                    value = 1 if winner == node.game_state.current_player else -1
            
            # Backpropagation
            node.backpropagate(value)
        
        # Extract top 3 moves
        top_moves = []
        for child in sorted(root.children, key=lambda c: c.N, reverse=True)[:3]:
            move = child.move
            n = child.N
            q = child.W / child.N if child.N > 0 else 0.0
            p = child.P
            # Calculate win rate: (W + N) / (2 * N) to normalize between 0 and 1
            win_rate = (child.W + n) / (2 * n) if n > 0 else 0.5
            top_moves.append((move, n, q, p, win_rate))
        
        self.top_moves = top_moves
        best_move = max(root.children, key=lambda c: c.N).move if root.children else None
        return best_move, root

    def search_mcts_with_stats(self, mcts, game, num_simulations):
        """Run MCTS search and return move + root node for statistics"""
        from othello_mcts import MCTSNode
        root = MCTSNode(game.clone())
        
        # Run simulations
        for _ in range(num_simulations):
            node = root
            
            # Selection: Traverse the tree using UCB1 until we reach a leaf
            while not node.is_terminal() and node.is_fully_expanded():
                next_node = node.best_child(mcts.exploration_weight)
                if next_node is None:
                    break
                node = next_node
            
            # Expansion: If the node is not terminal, expand it
            if not node.is_terminal() and not node.is_fully_expanded():
                expanded_node = node.expand()
                if expanded_node:
                    node = expanded_node
            
            # Simulation: Run a random playout from the node
            result = node.simulate()
            
            # Backpropagation: Update statistics up the tree
            node.backpropagate(result)
        
        # Extract top 3 moves
        top_moves = []
        for child in sorted(root.children, key=lambda c: c.visits, reverse=True)[:3]:
            move = child.move
            n = child.visits
            q = child.wins / child.visits if child.visits > 0 else 0.0
            p = child.wins / child.visits if child.visits > 0 else 0.0
            # Win rate is directly the wins/visits ratio
            win_rate = child.wins / child.visits if child.visits > 0 else 0.5
            top_moves.append((move, n, q, p, win_rate))
        
        self.top_moves = top_moves
        best_move = max(root.children, key=lambda c: c.visits).move if root.children else None
        return best_move, root

    def complete_ai_move(self, move, root_node):
        """Complete the AI move and update analysis panel"""
        self.make_move_with_animation(move)
        self.ai_running = False
        
        # Update analysis panel with the moves that were considered
        self.update_analysis_panel()
        
        # If MCTS vs MCTS is active, trigger next move
        if self.mcts_vs_mcts_active and not self.game.status()[0]:
            self.root.after(500, self.ai_move)

    def analyze_position(self):
        """Analyze current position and show top 3 moves"""
        if self.ai_running or self.game.status()[0]:
            messagebox.showwarning("Warning", "Cannot analyze now. Game is running or over.")
            return
        
        if not self.ai_algorithm:
            messagebox.showerror("Error", "AI algorithm not selected!")
            return
        
        # Disable analysis button during analysis
        self.btn_analysis.config(state="disabled")
        
        def run_analysis():
            try:
                if self.ai_algorithm == "PUCT":
                    _, _ = self.search_puct_with_stats(PUCTSearch(self.ai_net), self.game.clone(), num_simulations=200)
                elif self.ai_algorithm == "MCTS":
                    _, _ = self.search_mcts_with_stats(MCTS(), self.game.clone(), num_simulations=200)
                
                self.root.after(0, self.update_analysis_panel)
            except Exception as e:
                print(f"Error during analysis: {e}")
                import traceback
                traceback.print_exc()
                messagebox.showerror("Error", f"Analysis failed: {str(e)}")
            finally:
                self.root.after(0, lambda: self.btn_analysis.config(state="normal"))
        
        threading.Thread(target=run_analysis).start()

    def start_human_vs_mcts(self, color):
        """Start human vs AI game"""
        self.reset_game()
        self.human_player = color
        self.mcts_vs_human_active = True
        self.mcts_vs_mcts_active = False
        color_name = "BLACK" if color == Othello.BLACK else "WHITE"
        messagebox.showinfo("Mode", f"Game Started! You are {color_name}.")
        if self.game.current_player != self.human_player:
            self.root.after(500, self.ai_move)

    def start_mcts_vs_mcts(self):
        """Start AI vs AI game"""
        self.reset_game()
        self.mcts_vs_mcts_active = True
        self.mcts_vs_human_active = False
        messagebox.showinfo("Mode", f"AI vs AI Started! Using {self.ai_algorithm}!")
        self.ai_move()

    def reset_game(self):
        """Reset the game"""
        self.mcts_vs_mcts_active = False
        self.mcts_vs_human_active = False
        self.ai_running = False
        self.game = Othello()
        self.top_moves = []
        self.draw_board()
        self.update_analysis_panel()

    def undo_move(self):
        """Undo the last move"""
        if self.ai_running: return
        self.game.unmake_move()
        self.draw_board()

if __name__ == "__main__":
    root = tk.Tk()
    gui = OthelloGUI(root)
    root.mainloop()
