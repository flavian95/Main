
import tkinter as tk
from tkinter import messagebox
import copy
import time

# The 9x9 Sudoku board extracted from the provided image
INITIAL_BOARD = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

class SudokuSolver:
    def __init__(self, board):
        self.board = copy.deepcopy(board)
        self.states_explored = 0

    def is_valid(self, row, col, num):
        if num in self.board[row]: return False
        for r in range(9):
            if self.board[r][col] == num: return False
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if self.board[r][c] == num: return False
        return True

    def get_valid_moves(self, row, col):
        return [num for num in range(1, 10) if self.is_valid(row, col, num)]

    # --- NAIVE BACKTRACKING ---
    def find_empty_naive(self):
        for r in range(9):
            for c in range(9):
                if self.board[r][c] == 0:
                    return r, c
        return None

    def solve_naive(self):
        self.states_explored += 1
        empty = self.find_empty_naive()
        if not empty: return True
        row, col = empty

        for num in range(1, 10):
            if self.is_valid(row, col, num):
                self.board[row][col] = num
                if self.solve_naive(): return True
                self.board[row][col] = 0
        return False

    # --- HEURISTIC (MRV) ---
    def find_empty_mrv(self):
        min_moves, best_cell = 10, None
        for r in range(9):
            for c in range(9):
                if self.board[r][c] == 0:
                    moves = self.get_valid_moves(r, c)
                    if len(moves) < min_moves:
                        min_moves = len(moves)
                        best_cell = (r, c)
                        if min_moves <= 1: return best_cell
        return best_cell

    def solve_heuristic(self):
        self.states_explored += 1
        empty = self.find_empty_mrv()
        if not empty: return True
        row, col = empty

        for num in self.get_valid_moves(row, col):
            self.board[row][col] = num
            if self.solve_heuristic(): return True
            self.board[row][col] = 0
        return False


class SudokuGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sudoku AI Solver")
        self.root.geometry("450x600")
        self.root.configure(padx=20, pady=20)
        
        self.entries = [[None for _ in range(9)] for _ in range(9)]
        
        self.create_grid()
        self.create_controls()
        self.load_board(INITIAL_BOARD)

    def create_grid(self):
        """Creates the 9x9 grid with visually thicker borders for 3x3 subgrids."""
        main_frame = tk.Frame(self.root, bg="black", bd=2)
        main_frame.pack(pady=10)

        for block_row in range(3):
            for block_col in range(3):
                # Create a frame for each 3x3 block
                block = tk.Frame(main_frame, bg="black", bd=1)
                block.grid(row=block_row, column=block_col, padx=1, pady=1)
                
                # Fill the block with 3x3 entry widgets
                for r in range(3):
                    for c in range(3):
                        grid_row = block_row * 3 + r
                        grid_col = block_col * 3 + c
                        
                        entry = tk.Entry(block, width=3, font=("Helvetica", 18, "bold"), 
                                         justify="center", bd=1, relief="solid")
                        entry.grid(row=r, column=c, padx=1, pady=1, ipady=5)
                        self.entries[grid_row][grid_col] = entry

    def create_controls(self):
        """Creates the buttons and labels for results."""
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill="x", pady=10)

        # Buttons
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack()
        
        tk.Button(btn_frame, text="Solve Naive (No Heuristics)", command=lambda: self.run_solver("naive"), 
                  bg="#ffcccc", font=("Helvetica", 10, "bold")).grid(row=0, column=0, padx=5, pady=5)
        
        tk.Button(btn_frame, text="Solve with MRV Heuristic", command=lambda: self.run_solver("mrv"), 
                  bg="#ccffcc", font=("Helvetica", 10, "bold")).grid(row=0, column=1, padx=5, pady=5)
                  
        tk.Button(btn_frame, text="Reset Board", command=lambda: self.load_board(INITIAL_BOARD),
                  font=("Helvetica", 10)).grid(row=1, column=0, columnspan=2, pady=10)

        # Results Label
        self.result_label = tk.Label(control_frame, text="Ready to solve...", font=("Helvetica", 11))
        self.result_label.pack(pady=10)

    def load_board(self, board):
        """Loads a 2D array into the UI."""
        self.result_label.config(text="Ready to solve...", fg="black")
        for r in range(9):
            for c in range(9):
                self.entries[r][c].delete(0, tk.END)
                if board[r][c] != 0:
                    self.entries[r][c].insert(0, str(board[r][c]))
                    self.entries[r][c].config(fg="black", bg="#f0f0f0") # Pre-filled cells
                else:
                    self.entries[r][c].config(fg="blue", bg="white") # Empty cells

    def get_board_from_ui(self):
        """Reads the current state of the UI into a 2D array."""
        board = [[0 for _ in range(9)] for _ in range(9)]
        for r in range(9):
            for c in range(9):
                val = self.entries[r][c].get()
                if val.isdigit() and 1 <= int(val) <= 9:
                    board[r][c] = int(val)
                elif val != "":
                    messagebox.showerror("Error", f"Invalid input at row {r+1}, col {c+1}")
                    return None
        return board

    def run_solver(self, method):
        """Executes the chosen solving method and updates the UI."""
        current_board = self.get_board_from_ui()
        if not current_board: return

        self.result_label.config(text="Solving... Please wait.", fg="blue")
        self.root.update()

        solver = SudokuSolver(current_board)
        start_time = time.time()

        if method == "naive":
            success = solver.solve_naive()
            method_name = "Naive Backtracking"
        else:
            success = solver.solve_heuristic()
            method_name = "MRV Heuristic"

        elapsed_time = time.time() - start_time

        if success:
            # Update the UI with the solved numbers
            for r in range(9):
                for c in range(9):
                    if self.entries[r][c].get() == "":
                        self.entries[r][c].insert(0, str(solver.board[r][c]))
                        self.entries[r][c].config(fg="#006600") # Highlight solved numbers in green

            stats = f"{method_name} Success!\nStates Explored: {solver.states_explored}\nTime: {elapsed_time:.5f} sec"
            self.result_label.config(text=stats, fg="green")
        else:
            self.result_label.config(text="Unsolvable puzzle based on current inputs.", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = SudokuGUI(root)
    root.mainloop()