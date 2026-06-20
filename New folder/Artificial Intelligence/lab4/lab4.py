import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable

# ------------------------------
# 1. Data loading (TSPLIB format)
# ------------------------------

def read_tsp_file(filepath: str) -> Tuple[List[Tuple[float, float]], str]:
    """
    Read a TSPLIB .tsp file and return the list of city coordinates
    and the edge weight type.
    """
    with open(filepath) as f:
        lines = f.readlines()

    coords = []
    edge_weight_type = None
    reading_nodes = False
    for line in lines:
        line = line.strip()
        if line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[1].strip()
        elif line.startswith("NODE_COORD_SECTION"):
            reading_nodes = True
            continue
        elif reading_nodes:
            if line == "EOF" or line == "":
                break
            parts = line.split()
            if len(parts) >= 3:
                # TSPLIB node index x y
                x, y = float(parts[1]), float(parts[2])
                coords.append((x, y))
    return coords, edge_weight_type


def build_distance_matrix(coords: List[Tuple[float, float]], edge_weight_type: str) -> np.ndarray:
    """
    Build the distance matrix from city coordinates.
    Supports EUC_2D (rounded Euclidean) and ATT (pseudo-Euclidean).
    """
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i][j] = 0.0
                continue
            xi, yi = coords[i]
            xj, yj = coords[j]
            dx = xi - xj
            dy = yi - yj
            if edge_weight_type == "EUC_2D":
                # Rounded Euclidean distance
                dist[i][j] = round(math.sqrt(dx*dx + dy*dy))
            elif edge_weight_type == "ATT":
                # Pseudo-Euclidean
                rij = math.sqrt((dx*dx + dy*dy) / 10.0)
                tij = int(round(rij))
                dist[i][j] = tij if tij < rij else tij + 1
            else:
                # Fallback: exact Euclidean
                dist[i][j] = math.sqrt(dx*dx + dy*dy)
    return dist


# ------------------------------
# 2. Representation (permutation)
# ------------------------------

def route_length(route: List[int], dist: np.ndarray) -> float:
    """Compute total tour length (fitness)."""
    length = 0.0
    for i in range(len(route)):
        length += dist[route[i-1]][route[i]]
    return length


# ------------------------------
# 3. Neighbourhood operators (HC & SA)
# ------------------------------

def two_opt_swap(route: List[int], i: int, j: int) -> List[int]:
    """Apply a 2-opt move: reverse segment between i and j."""
    new_route = route[:]
    new_route[i:j+1] = reversed(route[i:j+1])
    return new_route


def random_two_opt_neighbor(route: List[int]) -> List[int]:
    """Generate a random 2-opt neighbour."""
    i, j = sorted(random.sample(range(len(route)), 2))
    return two_opt_swap(route, i, j)


# ------------------------------
# 4. Hill Climbing
# ------------------------------

def hill_climbing(dist: np.ndarray, max_evals: int = 10000) -> Tuple[List[int], float, List[float], int]:
    """
    Steepest ascent (or rather steepest descent) Hill Climbing.
    Returns: best route, its length, history of best lengths, and evaluation count when best was found.
    """
    n = len(dist)
    # Start from a random tour
    current = list(range(n))
    random.shuffle(current)
    current_len = route_length(current, dist)
    best_route = current[:]
    best_len = current_len
    history = [best_len]
    evals = 1
    best_at_eval = 1

    while evals < max_evals:
        # Examine all 2-opt neighbours (optional: could use random neighbourhood)
        # For efficiency, evaluate a random subset if n is large.
        # We use random neighbor for simplicity; for small instances exhaustive is possible.
        neighbor = random_two_opt_neighbor(current)
        neighbor_len = route_length(neighbor, dist)
        evals += 1

        if neighbor_len < current_len:
            current = neighbor
            current_len = neighbor_len
            if current_len < best_len:
                best_route = current[:]
                best_len = current_len
                best_at_eval = evals
        # If no improvement, stay with current (or can use multi-start)
        history.append(best_len)

    return best_route, best_len, history, best_at_eval


# ------------------------------
# 5. Simulated Annealing
# ------------------------------

def simulated_annealing(dist: np.ndarray,
                        init_temp: float = 1000.0,
                        cooling_rate: float = 0.995,
                        max_evals: int = 20000) -> Tuple[List[int], float, List[float], int]:
    """
    Simulated Annealing with 2-opt moves and exponential cooling.
    """
    n = len(dist)
    current = list(range(n))
    random.shuffle(current)
    current_len = route_length(current, dist)
    best_route = current[:]
    best_len = current_len
    history = [best_len]
    evals = 1
    best_at_eval = 1
    T = init_temp

    while evals < max_evals:
        neighbor = random_two_opt_neighbor(current)
        neighbor_len = route_length(neighbor, dist)
        evals += 1

        delta = neighbor_len - current_len
        if delta < 0 or random.random() < math.exp(-delta / T):
            current = neighbor
            current_len = neighbor_len
            if current_len < best_len:
                best_route = current[:]
                best_len = current_len
                best_at_eval = evals

        T *= cooling_rate
        history.append(best_len)

    return best_route, best_len, history, best_at_eval


# ------------------------------
# 6. Genetic Algorithm operators
# ------------------------------

def order_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """Order crossover (OX) for permutations."""
    size = len(parent1)
    # Choose two random cut points
    a, b = sorted(random.sample(range(size), 2))
    # Child 1
    child1 = [None] * size
    child1[a:b+1] = parent1[a:b+1]
    # Fill remaining with parent2 order
    p2_idx = 0
    for i in range(size):
        if child1[i] is None:
            while parent2[p2_idx] in child1:
                p2_idx += 1
            child1[i] = parent2[p2_idx]
            p2_idx += 1
    # Child 2
    child2 = [None] * size
    child2[a:b+1] = parent2[a:b+1]
    p1_idx = 0
    for i in range(size):
        if child2[i] is None:
            while parent1[p1_idx] in child2:
                p1_idx += 1
            child2[i] = parent1[p1_idx]
            p1_idx += 1
    return child1, child2


def swap_mutation(individual: List[int], mutation_rate: float = 0.1) -> List[int]:
    """Swap two randomly chosen cities with given probability."""
    mutated = individual[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def tournament_selection(population: List[List[int]], fitness: List[float], k: int = 3) -> List[int]:
    """Tournament selection: pick the best among k random individuals."""
    selected_idx = random.choices(range(len(population)), k=k)
    best_idx = min(selected_idx, key=lambda idx: fitness[idx])
    return population[best_idx][:]


def genetic_algorithm(dist: np.ndarray,
                      pop_size: int = 100,
                      generations: int = 500,
                      crossover_prob: float = 0.9,
                      mutation_prob: float = 0.1) -> Tuple[List[int], float, List[float], int]:
    """
    Steady-state GA with elitism (keep best 2).
    Returns best route, fitness, history of best fitness per generation, and generation when best was found.
    """
    n = len(dist)
    # Initial random population
    population = [list(range(n)) for _ in range(pop_size)]
    for ind in population:
        random.shuffle(ind)
    fitness = [route_length(ind, dist) for ind in population]
    best_idx = np.argmin(fitness)
    best_route = population[best_idx][:]
    best_fitness = fitness[best_idx]
    history = [best_fitness]
    best_gen = 0

    for gen in range(1, generations + 1):
        # Elitism: keep the two best individuals
        sorted_indices = np.argsort(fitness)
        new_population = [population[sorted_indices[0]][:], population[sorted_indices[1]][:]]

        while len(new_population) < pop_size:
            # Selection
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
            # Crossover
            if random.random() < crossover_prob:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            # Mutation
            child1 = swap_mutation(child1, mutation_prob)
            child2 = swap_mutation(child2, mutation_prob)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population
        fitness = [route_length(ind, dist) for ind in population]
        current_best = min(fitness)
        if current_best < best_fitness:
            best_fitness = current_best
            best_idx = np.argmin(fitness)
            best_route = population[best_idx][:]
            best_gen = gen
        history.append(best_fitness)

    # Number of evaluations: pop_size * generations
    return best_route, best_fitness, history, best_gen


# ------------------------------
# 7. Experiment runner and statistics
# ------------------------------

def run_experiment(algorithm: Callable, dist: np.ndarray, runs: int = 30,
                   **kwargs) -> dict:
    """
    Run a given algorithm 'runs' times and collect statistics.
    algorithm: function that returns (best_route, best_fitness, history, best_at_eval_or_gen)
    """
    best_fitnesses = []
    best_evals = []
    best_routes = []

    for _ in range(runs):
        route, fit, hist, best_at = algorithm(dist, **kwargs)
        best_fitnesses.append(fit)
        best_evals.append(best_at)
        best_routes.append(route)

    stats = {
        'best_fitness': best_fitnesses,
        'best_eval': best_evals,
        'mean_fitness': np.mean(best_fitnesses),
        'max_fitness': np.max(best_fitnesses),
        'min_fitness': np.min(best_fitnesses),
        'std_fitness': np.std(best_fitnesses),
        'mean_eval': np.mean(best_evals),
    }
    return stats

def plot_evolution(history: List[float], title: str, filename: str,
                   xlabel: str = "Evaluations/Generations"):
    """Plot fitness evolution and save to PNG."""
    if not history:
        print(f"Warning: empty history for '{title}'. Cannot plot.")
        return
    plt.figure()
    plt.plot(history, linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel("Tour length")
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename, dpi=150, bbox_inches='tight')   # Save to file
    plt.close()                                           # Close the figure to free memory
    print(f"Plot saved as: {filename}")


# ------------------------------
# 8. Main demonstration
# ------------------------------

if __name__ == "__main__":
    # ----- Load a TSP instance -----
    # Example: berlin52.tsp (download from TSPLIB)
    # Replace with your file path
    tsp_file = "berlin52.tsp"  # <-- change this
    coords, ewt = read_tsp_file(tsp_file)
    dist_matrix = build_distance_matrix(coords, ewt)
    num_cities = len(coords)
    print(f"Loaded {tsp_file} with {num_cities} cities.")

    # ----- Set parameters -----
    HC_MAX_EVALS = 5000
    SA_MAX_EVALS = 20000
    SA_INIT_TEMP = 1000.0
    SA_COOLING = 0.995
    GA_POP_SIZE = 100
    GA_GENERATIONS = 500
    GA_CX_PROB = 0.9
    GA_MUT_PROB = 0.1

    # ----- Single run for plots -----


    # --- Hill Climbing single run ---
    _, _, hc_hist, _ = hill_climbing(dist_matrix, HC_MAX_EVALS)
    plot_evolution(hc_hist, "Hill Climbing - fitness vs evaluations",
               "hill_climbing_evolution.png", "Evaluations")

# --- Simulated Annealing single run (corrected argument order) ---
    _, _, sa_hist, _ = simulated_annealing(dist_matrix,
                                       init_temp=SA_INIT_TEMP,
                                       cooling_rate=SA_COOLING,
                                       max_evals=SA_MAX_EVALS)
    plot_evolution(sa_hist, "Simulated Annealing - fitness vs evaluations",
               "simulated_annealing_evolution.png", "Evaluations")

# --- Genetic Algorithm single run ---
    _, _, ga_hist, _ = genetic_algorithm(dist_matrix, GA_POP_SIZE, GA_GENERATIONS,
                                     GA_CX_PROB, GA_MUT_PROB)
    plot_evolution(ga_hist, "Genetic Algorithm - best fitness per generation",
               "genetic_algorithm_evolution.png", "Generations")

    # ----- 30-run statistics -----
    print("\n========= 30-run statistics =========")
    RUNS = 30

    print("\nHill Climbing:")
    hc_stats = run_experiment(hill_climbing, dist_matrix, RUNS, max_evals=HC_MAX_EVALS)
    print(f"  Mean fitness: {hc_stats['mean_fitness']:.2f}, Std: {hc_stats['std_fitness']:.2f}")
    print(f"  Min: {hc_stats['min_fitness']:.2f}, Max: {hc_stats['max_fitness']:.2f}")
    print(f"  Mean evals to best: {hc_stats['mean_eval']:.1f}")

    print("\nSimulated Annealing:")
    sa_stats = run_experiment(simulated_annealing, dist_matrix, RUNS,
                              max_evals=SA_MAX_EVALS, init_temp=SA_INIT_TEMP,
                              cooling_rate=SA_COOLING)
    print(f"  Mean fitness: {sa_stats['mean_fitness']:.2f}, Std: {sa_stats['std_fitness']:.2f}")
    print(f"  Min: {sa_stats['min_fitness']:.2f}, Max: {sa_stats['max_fitness']:.2f}")
    print(f"  Mean evals to best: {sa_stats['mean_eval']:.1f}")

    print("\nGenetic Algorithm:")
    ga_stats = run_experiment(genetic_algorithm, dist_matrix, RUNS,
                              pop_size=GA_POP_SIZE, generations=GA_GENERATIONS,
                              crossover_prob=GA_CX_PROB, mutation_prob=GA_MUT_PROB)
    print(f"  Mean fitness: {ga_stats['mean_fitness']:.2f}, Std: {ga_stats['std_fitness']:.2f}")
    print(f"  Min: {ga_stats['min_fitness']:.2f}, Max: {ga_stats['max_fitness']:.2f}")
    print(f"  Mean generation to best: {ga_stats['mean_eval']:.1f}")