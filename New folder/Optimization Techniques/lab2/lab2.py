
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import TruncatedSVD

# ---------------------------------------------------------
# 1. Load Data (Subset of 20 persons out of 40)
# ---------------------------------------------------------
print("Downloading/Loading Olivetti Faces dataset...")
# The Olivetti dataset has 40 people, 10 images each (64x64 pixels)
faces = fetch_olivetti_faces()
X = faces.data
y = faces.target

# Filter to only the first 20 persons (targets 0 to 19)
mask = y < 20
X_subset = X[mask]  # Shape: (200 images, 4096 pixels)
y_subset = y[mask]

# Center the data by subtracting the mean face
mean_face = np.mean(X_subset, axis=0)
X_centered = X_subset - mean_face

n_samples, n_features = X_centered.shape

# ---------------------------------------------------------
# 2. Define Method 1: The Optimized Eigenfaces (L Matrix)
# ---------------------------------------------------------
def compute_eigenfaces_eig(X_c, k):
    """Computes k eigenfaces using the optimized L-matrix approach."""
    # L = X_c @ X_c.T (Shape: 200 x 200) - Notice X_c has images as rows here
    L = np.dot(X_c, X_c.T) 
    
    # Compute eigenvalues and eigenvectors of L
    eigenvalues, eigenvectors = np.linalg.eig(L)
    
    # Sort them in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Keep only the top k
    W = eigenvectors[:, :k]
    
    # Map back to original space to get the phantoms/eigenfaces
    # U = X_c.T @ W (Shape: 4096 x k)
    eigenfaces = np.dot(X_c.T, W)
    
    # Normalize the eigenfaces
    eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)
    
    return eigenfaces.T # Return shape (k, 4096) to match TruncatedSVD

# ---------------------------------------------------------
# 3. Plotting the Eigenfaces (k=20)
# ---------------------------------------------------------
k_plot = 20
print(f"\nComputing top {k_plot} eigenfaces for plotting...")
eigenfaces_custom = compute_eigenfaces_eig(X_centered, k_plot)

# Plotting a 4x5 grid
fig, axes = plt.subplots(4, 5, figsize=(10, 8), subplot_kw={'xticks':[], 'yticks':[]})
fig.suptitle(f"Top {k_plot} Eigenfaces/Phantoms (Custom Eig Method)", fontsize=16)

for i, ax in enumerate(axes.flat):
    # Reshape the 4096 flat array back to 64x64 image
    ax.imshow(eigenfaces_custom[i].real.reshape(64, 64), cmap='gray')
    ax.set_title(f"Phantom {i+1}")

plt.tight_layout()
plt.savefig("eigenfaces_grid.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 4. Benchmarking k = [20, 40, 60, 80, 100]
# ---------------------------------------------------------
k_values = [20, 40, 60, 80, 100]

print(f"\n{'k':<5} | {'Eig/L-Matrix Time (s)':<25} | {'TruncatedSVD Time (s)':<25}")
print("-" * 62)

for k in k_values:
    # Method 1: Eig on L matrix
    t0 = perf_counter()
    _ = compute_eigenfaces_eig(X_centered, k)
    t1 = perf_counter()
    time_eig = t1 - t0
    
    # Method 2: TruncatedSVD from sklearn
    t0 = perf_counter()
    svd = TruncatedSVD(n_components=k, algorithm='randomized', random_state=42)
    _ = svd.fit(X_centered)
    t1 = perf_counter()
    time_svd = t1 - t0
    
    print(f"{k:<5} | {time_eig:<25.6f} | {time_svd:<25.6f}")