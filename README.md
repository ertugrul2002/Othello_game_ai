# Othello Ultimate: AI-Powered Board Game

> A feature-rich desktop implementation of the classic strategy board game Othello (Reversi). This project integrates advanced Artificial Intelligence algorithms with an interactive analytical dashboard, allowing players to compete against smart agents or analyze their games in real-time.

## Features

- **Advanced AI Opponents**: Choose between state-of-the-art search algorithms:
  - **MCTS**: Monte Carlo Tree Search for deep, strategic simulation-based decision making.
  - **PUCT**: Polynomial Upper Confidence Trees (Predictive UCT) utilizing specialized heuristic priors.
- **Analytical Engine & Dashboard**: 
  - Real-time score tracking for both Black and White pieces.
  - **Top 3 Moves Showcase**: A dynamic side-panel that suggests and evaluates the best legal moves available.
  - **Analyze Mode**: Dive deep into any board state to study optimal patterns and branches.
- **Robust Control Panel**:
  - Full game state manipulation: **New Game**, **Undo Move**, and **Exit**.
  - Dynamic role swapping: **Play as Black**, **Play as White**, or witness an automated **AI vs AI** battle.

---

## Technical Stack

- **Platform**: Desktop (Windows/Linux)
- **Core Algorithms**: Monte Carlo Tree Search (MCTS), PUCT
- **Architecture**: Object-Oriented design separating core game logic, AI decision trees, and UI state management.

---

## Installation & Running the Game

Follow these simple steps to set up the environment and run the application locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/ertugrul2002/Othello_game_ai.git
cd othello_game
```

### 2. Launch the Application
Run the primary GUI module using the following terminal command:
```bash
python othello_gui_final.py
```

---

## Game Rules & Overview

Othello is a strategy board game played on an 8×8 uncheckered board. 
1. **Setup**: The game begins with 4 discs placed in a standard cross-pattern in the center.
2. **Gameplay**: Players take turns placing discs with their assigned color facing up. A move is made by trapping one or more of the opponent's discs between two of your own.
3. **Objective**: Flank the opponent's pieces to flip them to your color. The player with the majority of discs when no more legal moves can be made wins.

---

## Controls & Usage

### ⚙️ Control Panel Layout
- **New Game**: Flushes the board matrix and resets score counters to 2-2.
- **Undo Move**: Steps back the turn stack to correct mistakes or explore alternate tactical branches.
- **Play as Black / White**: Changes the active player's alignment mid-game or at start.
- **AI vs AI**: Triggers a simulation loop where two distinct AI instances (or the same algorithm) compete.
- **Analyze**: Activates the evaluation overlay to calculate strategic values for prospective board coordinates.
