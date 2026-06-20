
import csv
import time
import heapq
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from collections import deque

# ==========================================
# 1. DATA INITIALIZATION
# ==========================================
def create_data_files():
    """Generates the CSV files to simulate dynamic loading if they don't exist."""
    if not os.path.exists("distances.csv"):
        distances = [
            ("Oslo", "Helsinki", 970), ("Helsinki", "Stockholm", 400), ("Oslo", "Stockholm", 570),
            ("Stockholm", "Copenhagen", 522), ("Copenhagen", "Warsaw", 668), ("Warsaw", "Bucharest", 946),
            ("Bucharest", "Athens", 1300), ("Budapest", "Bucharest", 900), ("Budapest", "Belgrade", 316),
            ("Belgrade", "Sofia", 330), ("Rome", "Palermo", 1043), ("Palermo", "Athens", 907),
            ("Rome", "Milan", 681), ("Milan", "Budapest", 789), ("Vienna", "Budapest", 217),
            ("Vienna", "Munich", 458), ("Prague", "Vienna", 312), ("Prague", "Berlin", 354),
            ("Berlin", "Copenhagen", 743), ("Berlin", "Amsterdam", 648), ("Munich", "Lyon", 753),
            ("Lyon", "Paris", 481), ("Lyon", "Bordeaux", 542), ("Madrid", "Barcelona", 628),
            ("Madrid", "Lisbon", 638), ("Lisbon", "London", 2210), ("Barcelona", "Lyon", 644),
            ("Paris", "London", 414), ("London", "Dublin", 463), ("London", "Glasgow", 667),
            ("Glasgow", "Amsterdam", 711), ("Budapest", "Prague", 443), ("Barcelona", "Rome", 1471),
            ("Paris", "Bordeaux", 579), ("Glasgow", "Dublin", 306)
        ]
        with open("distances.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["CityA", "CityB", "Distance"])
            writer.writerows(distances)

    if not os.path.exists("heuristics.csv"):
        # Note: In the provided image, Bucharest is 0, meaning this is the distance TO Bucharest.
        heuristics = [
            ("Amsterdam", 2280), ("Athens", 1300), ("Barcelona", 2670), ("Belgrade", 630),
            ("Berlin", 1800), ("Bordeaux", 2100), ("Budapest", 900), ("Copenhagen", 2250),
            ("Dublin", 2530), ("Glasgow", 2470), ("Helsinki", 2820), ("Lisbon", 3950),
            ("London", 2590), ("Lyon", 1660), ("Madrid", 3300), ("Milan", 1750),
            ("Munich", 1600), ("Oslo", 2870), ("Palermo", 1280), ("Paris", 2970),
            ("Prague", 1490), ("Rome", 1140), ("Sofia", 390), ("Stockholm", 2890),
            ("Vienna", 1150), ("Warsaw", 946), ("Bucharest", 0) 
        ]
        with open("heuristics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["City", "Heuristic"])
            writer.writerows(heuristics)

# ==========================================
# 2. PROBLEM REPRESENTATION & STATES
# ==========================================
class State:
    """Represents a node in the search tree."""
    def __init__(self, city, parent=None, path_cost=0, heuristic=0):
        self.city = city
        self.parent = parent
        self.g = path_cost  # Cost from start to current node
        self.h = heuristic  # Estimated cost to goal
        self.f = self.g + self.h # Total estimated cost

    def __lt__(self, other):
        """Allows priority queue to sort by f-value."""
        return self.f < other.f

class MapProblem:
    """Handles loading the environment and determining transitions."""
    def __init__(self, distances_file, heuristics_file):
        self.graph = {}
        self.heuristics = {}
        self.cities = set()
        self.load_data(distances_file, heuristics_file)

    def load_data(self, distances_file, heuristics_file):
        """Dynamically loads map data from CSV files."""
        with open(distances_file, "r") as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                city_a, city_b, dist = row[0], row[1], int(row[2])
                self.cities.update([city_a, city_b])
                if city_a not in self.graph: self.graph[city_a] = []
                if city_b not in self.graph: self.graph[city_b] = []
                self.graph[city_a].append((city_b, dist))
                self.graph[city_b].append((city_a, dist)) # Two-way roads

        with open(heuristics_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.heuristics[row[0]] = int(row[1])

    def get_successors(self, current_city):
        """Returns valid transitions from the current state."""
        return self.graph.get(current_city, [])

    def get_heuristic(self, city):
        return self.heuristics.get(city, 0)

    def is_goal(self, state, goal_city):
        return state.city == goal_city

# ==========================================
# 3. SEARCH ALGORITHMS
# ==========================================
class SearchAlgorithms:
    def __init__(self, problem):
        self.problem = problem

    def reconstruct_path(self, state):
        """Backtracks to find the path taken."""
        path = []
        current = state
        while current:
            path.append(current.city)
            current = current.parent
        return path[::-1]

    def run_bfs(self, start_city, goal_city, max_states=10000):
        """Breadth-First Search (Uninformed)."""
        start_time = time.time()
        start_state = State(start_city)
        
        if self.problem.is_goal(start_state, goal_city):
            return self.reconstruct_path(start_state), 0, 0, 0, 1
            
        frontier = deque([start_state])
        explored = set()
        nodes_visited = 0
        max_memory_nodes = 0

        while frontier:
            max_memory_nodes = max(max_memory_nodes, len(frontier) + len(explored))
            if nodes_visited >= max_states:
                break

            current_state = frontier.popleft()
            explored.add(current_state.city)
            nodes_visited += 1

            for neighbor_city, cost in self.problem.get_successors(current_state.city):
                if neighbor_city not in explored and not any(n.city == neighbor_city for n in frontier):
                    child_state = State(neighbor_city, current_state, current_state.g + cost)
                    
                    if self.problem.is_goal(child_state, goal_city):
                        execution_time = time.time() - start_time
                        path = self.reconstruct_path(child_state)
                        return path, child_state.g, nodes_visited, execution_time, max_memory_nodes
                        
                    frontier.append(child_state)

        return None, 0, nodes_visited, time.time() - start_time, max_memory_nodes

    def run_dfs(self, start_city, goal_city, max_states=10000):
        """Depth-First Search (Uninformed)."""
        start_time = time.time()
        start_state = State(start_city)
        
        # Using a list as a LIFO Stack
        frontier = [start_state]
        explored = set()
        nodes_visited = 0
        max_memory_nodes = 0

        while frontier:
            max_memory_nodes = max(max_memory_nodes, len(frontier) + len(explored))
            if nodes_visited >= max_states:
                break

            current_state = frontier.pop() # LIFO behavior
            
            if self.problem.is_goal(current_state, goal_city):
                execution_time = time.time() - start_time
                return self.reconstruct_path(current_state), current_state.g, nodes_visited, execution_time, max_memory_nodes
                
            if current_state.city not in explored:
                explored.add(current_state.city)
                nodes_visited += 1

                for neighbor_city, cost in reversed(self.problem.get_successors(current_state.city)):
                    if neighbor_city not in explored:
                        child_state = State(neighbor_city, current_state, current_state.g + cost)
                        frontier.append(child_state)

        return None, 0, nodes_visited, time.time() - start_time, max_memory_nodes

    def run_gbfs(self, start_city, goal_city, max_states=10000):
        """Greedy Best-First Search (Informed - ignores path cost, only uses heuristic)."""
        start_time = time.time()
        start_state = State(start_city, heuristic=self.problem.get_heuristic(start_city))
        
        frontier = []
        # Push a tuple (heuristic, id, state) so heapq sorts strictly by heuristic
        heapq.heappush(frontier, (start_state.h, id(start_state), start_state))
        explored = set()
        nodes_visited = 0
        max_memory_nodes = 0

        while frontier:
            max_memory_nodes = max(max_memory_nodes, len(frontier) + len(explored))
            if nodes_visited >= max_states:
                break

            _, _, current_state = heapq.heappop(frontier)
            
            if self.problem.is_goal(current_state, goal_city):
                execution_time = time.time() - start_time
                return self.reconstruct_path(current_state), current_state.g, nodes_visited, execution_time, max_memory_nodes
                
            if current_state.city not in explored:
                explored.add(current_state.city)
                nodes_visited += 1

                for neighbor_city, cost in self.problem.get_successors(current_state.city):
                    if neighbor_city not in explored:
                        heuristic = self.problem.get_heuristic(neighbor_city)
                        child_state = State(neighbor_city, current_state, current_state.g + cost, heuristic)
                        heapq.heappush(frontier, (child_state.h, id(child_state), child_state))

        return None, 0, nodes_visited, time.time() - start_time, max_memory_nodes

    def run_a_star(self, start_city, goal_city, max_states=10000):
        """A* Search (Informed)."""
        start_time = time.time()
        start_state = State(start_city, heuristic=self.problem.get_heuristic(start_city))
        
        frontier = []
        heapq.heappush(frontier, start_state)
        cost_so_far = {start_city: 0}
        nodes_visited = 0
        max_memory_nodes = 0

        while frontier:
            max_memory_nodes = max(max_memory_nodes, len(frontier) + len(cost_so_far))
            if nodes_visited >= max_states:
                break

            current_state = heapq.heappop(frontier)
            nodes_visited += 1

            if self.problem.is_goal(current_state, goal_city):
                execution_time = time.time() - start_time
                return self.reconstruct_path(current_state), current_state.g, nodes_visited, execution_time, max_memory_nodes

            for neighbor_city, cost in self.problem.get_successors(current_state.city):
                new_cost = current_state.g + cost
                
                if neighbor_city not in cost_so_far or new_cost < cost_so_far[neighbor_city]:
                    cost_so_far[neighbor_city] = new_cost
                    heuristic = self.problem.get_heuristic(neighbor_city)
                    child_state = State(neighbor_city, current_state, new_cost, heuristic)
                    heapq.heappush(frontier, child_state)

        return None, 0, nodes_visited, time.time() - start_time, max_memory_nodes

    def run_csp_backtracking(self, start_city, goal_city, max_states=10000):
        """Backtracking for CSP with Forward Checking (Adapted for Pathfinding)."""
        start_time = time.time()
        visited = {start_city}
        
        # Use mutable lists so they can be updated inside the nested recursive function
        nodes_visited = [0] 
        max_memory_nodes = [1]
        
        def backtrack(current_city, current_cost):
            nodes_visited[0] += 1
            if current_city == goal_city:
                return [current_city], current_cost
            if nodes_visited[0] >= max_states:
                return None, 0
                
            neighbors = self.problem.get_successors(current_city)
            valid_neighbors = []
            
            # Forward Checking phase
            for n_city, cost in neighbors:
                if n_city not in visited:
                    if n_city == goal_city:
                        valid_neighbors.append((n_city, cost))
                    else:
                        # Forward Check: Does this neighbor have at least one unvisited connection?
                        future_moves = [fn for fn, _ in self.problem.get_successors(n_city) if fn not in visited and fn != current_city]
                        if future_moves:
                            valid_neighbors.append((n_city, cost))
                            
            # Sort neighbors by heuristic (Value Ordering Heuristic) to improve efficiency
            valid_neighbors.sort(key=lambda x: self.problem.get_heuristic(x[0]))
            
            # Backtracking phase
            for n_city, cost in valid_neighbors:
                visited.add(n_city)
                max_memory_nodes[0] = max(max_memory_nodes[0], len(visited))
                
                path_suffix, total_cost = backtrack(n_city, current_cost + cost)
                if path_suffix is not None:
                    return [current_city] + path_suffix, total_cost
                    
                # Backtrack if it resulted in a dead-end
                visited.remove(n_city)
            
            return None, 0

        path, cost = backtrack(start_city, 0)
        return path, cost, nodes_visited[0], time.time() - start_time, max_memory_nodes[0]

# ==========================================
# 4. GRAPHICAL USER INTERFACE (Tkinter)
# ==========================================
class AppGUI:
    def __init__(self, root, problem, search_agent):
        self.root = root
        self.problem = problem
        self.search_agent = search_agent
        
        self.root.title("European City Route Finder (AI Search)")
        self.root.geometry("750x600") # Increased size slightly to fit new buttons
        self.root.configure(padx=20, pady=20)
        
        # Sort cities alphabetically for dropdowns
        self.city_list = sorted(list(self.problem.cities))
        
        self.create_widgets()

    def create_widgets(self):
        # --- Top Frame: Selection ---
        selection_frame = ttk.LabelFrame(self.root, text=" Route Configuration ", padding=10)
        selection_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(selection_frame, text="Start City:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.start_combo = ttk.Combobox(selection_frame, values=self.city_list, state="readonly", width=20)
        self.start_combo.set("London") # Default per assignment context
        self.start_combo.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(selection_frame, text="Goal City:").grid(row=0, column=2, padx=15, pady=5, sticky=tk.W)
        self.goal_combo = ttk.Combobox(selection_frame, values=self.city_list, state="readonly", width=20)
        self.goal_combo.set("Bucharest")
        self.goal_combo.grid(row=0, column=3, padx=5, pady=5)

        # --- Middle Frame: Action Buttons ---
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Reorganized buttons into a clean grid layout
        self.btn_bfs = ttk.Button(button_frame, text="BFS (Uninformed)", command=self.execute_bfs)
        self.btn_bfs.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        
        self.btn_dfs = ttk.Button(button_frame, text="DFS (Uninformed)", command=self.execute_dfs)
        self.btn_dfs.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.btn_gbfs = ttk.Button(button_frame, text="GBFS (Informed)", command=self.execute_gbfs)
        self.btn_gbfs.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        
        self.btn_astar = ttk.Button(button_frame, text="A* (Informed Optimal)", command=self.execute_astar)
        self.btn_astar.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        
        self.btn_csp = ttk.Button(button_frame, text="CSP (Backtrack + FC)", command=self.execute_csp)
        self.btn_csp.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.btn_clear = ttk.Button(button_frame, text="Clear Output", command=self.clear_output)
        self.btn_clear.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        # Configure columns to stretch evenly
        for i in range(3):
            button_frame.columnconfigure(i, weight=1)

        # --- Bottom Frame: Results Output ---
        output_frame = ttk.LabelFrame(self.root, text=" Search Results & Metrics ", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.text_output = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, font=("Consolas", 10))
        self.text_output.pack(fill=tk.BOTH, expand=True)
        
        # Display a welcome message
        self.log_message("Welcome to the AI Search Route Finder.\nSelect your cities and click an algorithm to begin.\n" + "-"*50)

    def get_cities(self):
        start = self.start_combo.get()
        goal = self.goal_combo.get()
        if not start or not goal:
            messagebox.showwarning("Input Error", "Please select both a Start and Goal city.")
            return None, None
        return start, goal

    def log_message(self, message):
        """Appends text to the output area."""
        self.text_output.insert(tk.END, message + "\n")
        self.text_output.see(tk.END) # Scroll to bottom

    def clear_output(self):
        self.text_output.delete('1.0', tk.END)

    def format_results(self, algo_name, path, cost, visited, exec_time, mem):
        if not path:
            return f"[{algo_name}] No path found between selected cities.\n"
        
        route_str = " -> ".join(path)
        transitions = len(path) - 1
        
        result = (
            f"=== {algo_name} ===\n"
            f"Path: {route_str}\n"
            f"Solution Length: {transitions} transitions\n"
            f"Total Distance: {cost} km\n"
            f"Explored States: {visited} nodes\n"
            f"Execution Time: {exec_time:.6f} seconds\n"
            f"Peak Memory: {mem} nodes stored\n"
            f"{'-'*50}"
        )
        return result

    def execute_bfs(self):
        start, goal = self.get_cities()
        if start and goal:
            self.log_message(f"Running BFS from {start} to {goal}...")
            path, cost, visited, exec_time, mem = self.search_agent.run_bfs(start, goal)
            self.log_message(self.format_results("Breadth-First Search", path, cost, visited, exec_time, mem))

    def execute_dfs(self):
        start, goal = self.get_cities()
        if start and goal:
            self.log_message(f"Running DFS from {start} to {goal}...")
            path, cost, visited, exec_time, mem = self.search_agent.run_dfs(start, goal)
            self.log_message(self.format_results("Depth-First Search", path, cost, visited, exec_time, mem))

    def execute_gbfs(self):
        start, goal = self.get_cities()
        if start and goal:
            self.log_message(f"Running GBFS from {start} to {goal}...")
            path, cost, visited, exec_time, mem = self.search_agent.run_gbfs(start, goal)
            self.log_message(self.format_results("Greedy Best-First Search", path, cost, visited, exec_time, mem))

    def execute_astar(self):
        start, goal = self.get_cities()
        if start and goal:
            self.log_message(f"Running A* from {start} to {goal}...")
            path, cost, visited, exec_time, mem = self.search_agent.run_a_star(start, goal)
            self.log_message(self.format_results("A* Search", path, cost, visited, exec_time, mem))

    def execute_csp(self):
        start, goal = self.get_cities()
        if start and goal:
            self.log_message(f"Running CSP Backtracking (with FC) from {start} to {goal}...")
            path, cost, visited, exec_time, mem = self.search_agent.run_csp_backtracking(start, goal)
            self.log_message(self.format_results("CSP Backtracking (Forward Checking)", path, cost, visited, exec_time, mem))


# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Create data files if missing
    create_data_files()
    
    # 2. Initialize problem and search agent
    problem = MapProblem("distances.csv", "heuristics.csv")
    search_agent = SearchAlgorithms(problem)
    
    # 3. Launch GUI
    root = tk.Tk()
    app = AppGUI(root, problem, search_agent)
    root.mainloop()
