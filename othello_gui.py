import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import time
import threading
from othello import Othello
import threading
from othello_network import GameNetworkSimple
from othello_puct import PUCTSearch
from othello_mcts import MCTS

class OthelloGUI:
    def __init__(self, root):
        # Initialize the main window and game logic
        self.root = root
        self.root.title("Othello Ultimate with MCTS")
        self.root.configure(bg="#1e1e1e")
        self.game = Othello()
        # self.mcts = MCTS()
        self.ai_net = GameNetworkSimple(action_size=64)
        try:
            # تأكد من كتابة اسم ملف الموديل الذي حفظته بعد التدريب
            self.ai_net.model.load_weights('my_othello_model_8000_v5.keras')
            print("Model weights loaded successfully! 8000_v5")
            # self.ai_net.model.load_weights('my_othello_model_1000_v1.h5')
            # print("Model weights loaded successfully! 1000")
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
        
        self.load_assets()
        self.setup_ui()
        self.draw_board()
        self.animating = False

    def load_assets(self):
        # Load and resize piece images
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
        # Setup the layout with sidebar on the left and board on the right
        self.main_container = tk.Frame(self.root, bg=self.BG_COLOR)
        self.main_container.pack(expand=True, fill="both", padx=20, pady=20)
        
        # Sidebar (Left)
        self.sidebar = tk.Frame(self.main_container, bg=self.SIDEBAR_COLOR, width=220, padx=15, pady=20)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        
        tk.Label(self.sidebar, text="CONTROL PANEL", font=("Segoe UI", 14, "bold"), bg=self.SIDEBAR_COLOR, fg="white").pack(pady=(0, 20), fill="x")
        
        btn_style = {"font": ("Segoe UI", 10, "bold"), "fg": "white", "pady": 10, "relief": "flat", "cursor": "hand2"}
        tk.Button(self.sidebar, text="NEW GAME", command=self.reset_game, bg="#2ecc71", **btn_style).pack(fill="x", pady=5)
        tk.Button(self.sidebar, text="UNDO MOVE", command=self.undo_move, bg="#e74c3c", **btn_style).pack(fill="x", pady=5)
        
        self.btn_play_black = tk.Button(self.sidebar, text="PLAY AS BLACK", command=lambda: self.start_human_vs_mcts(Othello.BLACK), bg="#3498db", **btn_style)
        self.btn_play_black.pack(fill="x", pady=5)

        self.btn_play_white = tk.Button(self.sidebar, text="PLAY AS WHITE", command=lambda: self.start_human_vs_mcts(Othello.WHITE), bg="#9b59b6", **btn_style)
        self.btn_play_white.pack(fill="x", pady=5)
        
        self.btn_mcts_mcts = tk.Button(self.sidebar, text="MCTS vs MCTS", command=self.start_mcts_vs_mcts, bg="#95a5a6", **btn_style)
        self.btn_mcts_mcts.pack(fill="x", pady=5)
        
        tk.Frame(self.sidebar, bg=self.SIDEBAR_COLOR).pack(expand=True, fill="both")
        tk.Button(self.sidebar, text="EXIT", command=self.root.quit, bg="#34495e", **btn_style).pack(fill="x", pady=(10, 0))
        
        # Board Area (Right)
        self.board_area = tk.Frame(self.main_container, bg=self.BG_COLOR)
        self.board_area.pack(side="right", padx=20)
        
        # Current Turn Frame (Top)
        self.turn_frame = tk.Frame(self.board_area, bg=self.SIDEBAR_COLOR, padx=20, pady=10, highlightbackground="#333333", highlightthickness=1)
        self.turn_frame.pack(fill="x", pady=(0, 15))
        
        self.turn_title = tk.Label(self.turn_frame, text="CURRENT TURN: ", font=("Segoe UI", 12, "bold"), bg=self.SIDEBAR_COLOR, fg="#2ecc71")
        self.turn_title.pack(side="left")
        
        self.turn_value = tk.Label(self.turn_frame, text="BLACK", font=("Segoe UI", 12, "bold"), bg=self.SIDEBAR_COLOR)
        self.turn_value.pack(side="left")
        
        # The Canvas
        self.canvas = tk.Canvas(self.board_area, width=480, height=480, bg=self.BOARD_COLOR, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Score Frame (Bottom)
        self.score_container = tk.Frame(self.board_area, bg=self.SIDEBAR_COLOR, padx=20, pady=10, highlightbackground="#545353", highlightthickness=1)
        self.score_container.pack(fill="x", pady=(15, 0))

        tk.Label(self.score_container, text="SCORE: ", font=("Segoe UI", 11, "bold"), bg=self.SIDEBAR_COLOR, fg="#2ecc71").pack(side="left")

        self.black_score_label = tk.Label(self.score_container, text="BLACK: 2", font=("Segoe UI", 11, "bold"), bg=self.SIDEBAR_COLOR, fg="#aaaaaa")
        self.black_score_label.pack(side="left", padx=(20, 40))
        
        self.white_score_label = tk.Label(self.score_container, text="WHITE: 2", font=("Segoe UI", 11, "bold"), bg=self.SIDEBAR_COLOR, fg="white")
        self.white_score_label.pack(side="left")

    def animate_flip(self, r, c, to_player):
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
        self.canvas.delete("all")
        for i in range(9):
            self.canvas.create_line(i*60, 0, i*60, 480, fill=self.GRID_COLOR)
            self.canvas.create_line(0, i*60, 480, i*60, fill=self.GRID_COLOR)
        
        legal_moves = self.game.legal_moves()
        for r, c in legal_moves:
            self.canvas.create_oval(c*60+25, r*60+25, c*60+35, r*60+35, fill="white", outline="")
            
        for r in range(8):
            for c in range(8):
                piece = self.game.board[r][c]
                if piece == Othello.BLACK:
                    self.canvas.create_image(c*60+30, r*60+30, image=self.black_photo, tags=f"piece_{r}_{c}")
                elif piece == Othello.WHITE:
                    self.canvas.create_image(c*60+30, r*60+30, image=self.white_photo, tags=f"piece_{r}_{c}")
        self.update_info()

    def update_info(self):
        if self.game.current_player == Othello.BLACK:
            self.turn_value.config(text="BLACK", fg="#aaaaaa")
        else:
            self.turn_value.config(text="WHITE", fg="white")
            
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
            else: msg += "It\'s a Draw!"
            messagebox.showinfo("Game Over", msg)

    def make_move_with_animation(self, move):
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
        
        # # Check if next player has moves, if not, pass
        # if not self.game.legal_moves() and not self.game.status()[0]:
        #     self.root.after(500, lambda: self.make_move_with_animation(None))
        # elif self.mcts_vs_human_active and self.game.current_player != self.human_player and not self.game.status()[0]:
        #     # If it's AI's turn and it has moves, trigger AI move
        #     self.root.after(500, self.ai_move)
        is_over, _ = self.game.status()
        if is_over:
            self.update_info() # لتحديث النتيجة النهائية وإظهار رسالة الفوز
            return

        legal_moves = self.game.legal_moves()
        
        if not legal_moves:
            # إذا لم يوجد حركات قانونية للاعب الحالي، يجب تمرير الدور
            player_name = "Black" if self.game.current_player == Othello.BLACK else "White"
            messagebox.showinfo("Pass", f"{player_name} has no legal moves. Passing turn...")
            # استدعاء الدالة بقيمة None لتغيير الدور في الميكانيك الداخلي
            self.root.after(500, lambda: self.make_move_with_animation(None))
            return

        # تشغيل الذكاء الاصطناعي فقط في الحالات التالية:
        # 1. إذا كان نمط MCTS vs Human نشطاً والدور ليس دور البشر
        if self.mcts_vs_human_active and self.game.current_player != self.human_player:
            self.root.after(500, self.ai_move)
            
        # 2. إذا كان نمط MCTS vs MCTS نشطاً
        elif self.mcts_vs_mcts_active:
            self.root.after(500, self.ai_move)

    def on_click(self, event):
        if self.animating or self.ai_running: return
        if self.mcts_vs_mcts_active: return
        if self.mcts_vs_human_active and self.game.current_player != self.human_player: return
        
        c, r = event.x // 60, event.y // 60
        if (r, c) in self.game.legal_moves():
            self.make_move_with_animation((r, c))
            
            # If MCTS vs Human is active and it\'s AI\'s turn
            if self.mcts_vs_human_active and self.game.current_player != self.human_player:
                self.root.after(500, self.ai_move)

    # def get_nn_move(self):
    #     # 1. تحويل اللوحة الحالية إلى مصفوفة Numpy
    #     board_np = np.array(self.game.board)
        
    #     # 2. تجهيز القنوات الثلاث (لاعب، خصم، دور)
    #     # القناة 1: قطع اللاعب الأسود
    #     chan_black = (board_np == Othello.BLACK).astype(np.float32)
    #     # القناة 2: قطع اللاعب الأبيض
    #     chan_white = (board_np == Othello.WHITE).astype(np.float32)
    #     # القناة 3: دور اللاعب (1 إذا أسود، 0 إذا أبيض)
    #     turn_val = 1.0 if self.game.current_player == Othello.BLACK else 0.0
    #     chan_turn = np.full((8, 8), turn_val, dtype=np.float32)
        
    #     # 3. دمج القنوات بتنسيق (3, 8, 8) كما طلبت
    #     state = np.stack([chan_black, chan_white, chan_turn], axis=0)
        
    #     # 4. إضافة بُعد الـ Batch ليصبح (1, 3, 8, 8)
    #     state = np.expand_dims(state, axis=0)
        
    #     # 5. التوقع من الموديل
    #     policy, value = self.ai_net.model.predict(state, verbose=0)
    #     policy = policy[0] # نأخذ أول نتيجة من الـ Batch
        
    #     # 6. تصفية الحركات القانونية فقط لضمان عدم الخطأ
    #     legal_moves = self.game.legal_moves()
    #     best_move = None
    #     max_prob = -1.0
        
    #     for r, c in legal_moves:
    #         idx = r * 8 + c
    #         if policy[idx] > max_prob:
    #             max_prob = policy[idx]
    #             best_move = (r, c)
                
    #     return best_move

    # def ai_move(self):
    #     if self.ai_running or self.game.status()[0]: return
    #     self.ai_running = True
    #     move = self.get_nn_move()
        
    #     # تنفيذ الحركة في الواجهة
    #     self.root.after(500, lambda: self.complete_ai_move(move))

    #     # def run_mcts():
    #     #     self.ai_running = True
    #     #     move = self.mcts.search(self.game, num_simulations=1000)
    #     #     self.root.after(0, lambda: self.complete_ai_move(move))
            
    #     # threading.Thread(target=run_mcts).start()

    def ai_move(self):
        if self.mcts_vs_human_active and self.game.current_player == self.human_player:
            return
        
        if self.ai_running or self.game.status()[0]: return
        self.ai_running = True
        
        def run_puct():
            # نحن هنا نمرر self.ai_net (الذي يحتوي على موديلك المدرب) للـ Search
            # الـ Search بدون هذا الموديل لا يستطيع العمل
            searcher = PUCTSearch(self.ai_net) 
            
            # الموديل سيعطي "رأيه" والـ Search سيقوم بتجربة هذا الرأي 800 مرة
            move = searcher.search(self.game, num_simulations=200)
            
            self.root.after(0, lambda: self.complete_ai_move(move))
            
        threading.Thread(target=run_puct).start()

    def complete_ai_move(self, move):
        self.make_move_with_animation(move)
        self.ai_running = False
        
        # If MCTS vs MCTS is active, trigger next move
        if self.mcts_vs_mcts_active and not self.game.status()[0]:
            self.root.after(500, self.ai_move)

    def start_human_vs_mcts(self, color):
        self.reset_game()
        self.human_player = color
        self.mcts_vs_human_active = True
        self.mcts_vs_mcts_active = False
        color_name = "BLACK" if color == Othello.BLACK else "WHITE"
        messagebox.showinfo("Mode", f"Game Started! You are {color_name}.")
        if self.game.current_player != self.human_player:
            self.root.after(500, self.ai_move)

    def start_mcts_vs_mcts(self):
        self.reset_game()
        self.mcts_vs_mcts_active = True
        self.mcts_vs_human_active = False
        messagebox.showinfo("Mode", "MCTS vs MCTS Started!")
        self.ai_move()

    def reset_game(self):
        self.mcts_vs_mcts_active = False
        self.mcts_vs_human_active = False
        self.ai_running = False
        self.game = Othello()
        self.draw_board()

    def undo_move(self):
        if self.ai_running: return
        self.game.unmake_move()
        self.draw_board()

if __name__ == "__main__":
    root = tk.Tk()
    gui = OthelloGUI(root)
    root.mainloop()