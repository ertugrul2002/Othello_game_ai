import numpy as np
import copy
import math


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


class Othello:
    BLACK = 1
    WHITE = 2
    EMPTY = 0

    WHITE_WIN = 1
    BLACK_WIN = -1
    DRAW = 0
    ONGOING = -17 

    def __init__(self):
        """
        init: intialize a new Othello game with the starting position.
        8*8 board with initial four pieces in the center.
        1 represents black pieces, 2 represents white pieces, and 0 represents empty squares.
        1 (black) starts first.

        """
        self.board = [[self.EMPTY for _ in range(8)] for _ in range(8)]
        # add initial four pieces
        self.board[3][3] = self.WHITE
        self.board[3][4] = self.BLACK
        self.board[4][3] = self.BLACK
        self.board[4][4] = self.WHITE
        # self.player = self.BLACK
        self.history = []
        self.current_player = self.BLACK
        

    def legal_moves(self):
        """
        legal_moves: return a list of all legal moves for the current player.
        Each move is represented as a tuple (row, column).
        """
        player = self.current_player
        moves = []
        for r in range(8):
            for c in range(8):
                if self._is_valid_move(r, c, player):
                    moves.append((r, c))
        return moves

    def _is_valid_move(self, r, c, player):
        if self.board[r][c] != self.EMPTY:
            return False
        
        opponent = 3 - player
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr][nc] == opponent:
                nr += dr
                nc += dc
                while 0 <= nr < 8 and 0 <= nc < 8:
                    if self.board[nr][nc] == self.EMPTY:
                        break
                    if self.board[nr][nc] == player:
                        return True
                    nr += dr
                    nc += dc
        return False

    def make_move(self, move):
        """
        make_move : simulate making a move on the board for the current player.
        move: tuple (row, column) indicating where to place the piece.
        3 - current_player to switch turns.
        """
        if move is None: # pass move if no legal moves
            self.history.append((copy.deepcopy(self.board), self.current_player))
            self.current_player = self.WHITE if self.current_player == self.BLACK else self.BLACK
            return

        r, c = move
       
        self.history.append((copy.deepcopy(self.board), self.current_player))
        
        self.board[r][c] = self.current_player
        opponent = self.WHITE if self.current_player == self.BLACK else self.BLACK

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dr, dc in directions:
            to_flip = []
            nr, nc = r + dr, c + dc
            while 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr][nc] == opponent:
                to_flip.append((nr, nc))
                nr += dr
                nc += dc
            
            if 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr][nc] == self.current_player:
                for fr, fc in to_flip:
                    self.board[fr][fc] = self.current_player
        
        self.current_player = self.WHITE if self.current_player == self.BLACK else self.BLACK

    def unmake_move(self):
        """
        unmake_move: التراجع عن آخر حركة واستعادة الحالة السابقة.
        """
        if not self.history:
            return
        
        prev_board, prev_player = self.history.pop()
        self.board = prev_board
        self.current_player = prev_player

    def clone(self):
        """
        clone: إنشاء نسخة عميقة من كائن اللعبة الحالي.
        """
        new_game = Othello()
        new_game.board = copy.deepcopy(self.board)
        new_game.current_player = self.current_player
        new_game.history = copy.deepcopy(self.history)
        return new_game

    def encode(self):
        my_pieces = [
            [1.0 if cell == self.current_player else 0.0 for cell in row] 
            for row in self.board
        ]
        opponent = 2 if self.current_player == 1 else 1
        opp_pieces = [
            [1.0 if cell == opponent else 0.0 for cell in row] 
            for row in self.board
        ]

        turn_val = 1.0 if self.current_player == 1 else 0.0
        turn_plane = [[turn_val for _ in range(8)] for _ in range(8)]

        # return np.array([my_pieces, opp_pieces, turn_plane], dtype=np.float32)
        return np.transpose(np.array([my_pieces, opp_pieces, turn_plane], dtype=np.float32),(1, 2, 0))


    def decode(self, data):
        """
        decode: استعادة حالة اللعبة من البيانات المرمزة.
        """
        self.current_player = data[-1]
        board_data = data[:-1]
        for i in range(8):
            self.board[i] = board_data[i*8 : (i+1)*8]
        self.history = [] # تصفير التاريخ عند التحميل من ترميز خارجي

    def status(self):
        """
        Determines the game status.
        Returns: (is_over, winner)
        Winner: 1 for Black, 2 for White, 0 for Draw, None if not over.
        """
        # Check if there are any legal moves for current player
        current_moves = self.legal_moves()
        
        # Check if there are any legal moves for opponent
        original_player = self.current_player
        self.current_player = 3 - original_player
        opponent_moves = self.legal_moves()
        self.current_player = original_player
        
        # If neither player has legal moves, the game is over
        if not current_moves and not opponent_moves:
            black_score = sum(row.count(self.BLACK) for row in self.board)
            white_score = sum(row.count(self.WHITE) for row in self.board)
            
            if black_score > white_score:
                return True, self.BLACK
            elif white_score > black_score:
                return True, self.WHITE
            else:
                return True, self.DRAW # Draw
                
        return False, None

    def display(self):
        """display the current board state."""
        print("  0 1 2 3 4 5 6 7")
        for i, row in enumerate(self.board):
            chars = {self.EMPTY: '.', self.BLACK: 'B', self.WHITE: 'W'}
            print(f"{i} {' '.join(chars[cell] for cell in row)}")
        print(f"Current Player: {'Black' if self.current_player == self.BLACK else 'White'}")

    def get_heuristic_value(self):
        """تحسب تقييم يدوي للوحة الحالية بناءً على أماكن القطع"""
        score = 0
        for r in range(8):
            for c in range(8):
                if self.board[r][c] == self.current_player:
                    score += WEIGHTS[r][c]
                elif self.board[r][c] == (3 - self.current_player):
                    score -= WEIGHTS[r][c]
        
        # تحويل النتيجة لكسر بين -1 و 1 باستخدام tanh
        return math.tanh(score / 100.0)

