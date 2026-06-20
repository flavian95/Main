# =========================================
# CASE 2: MULTIPLE REGRESSION (MANUAL)
# =========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def build_poly_features_2d(X, degree):
    """Manually builds polynomial features for 2 variables up to 'degree'"""
    x1 = X[:, 0]
    x2 = X[:, 1]
    cols = [np.ones(len(x1))] # Bias column
    for d in range(1, degree + 1):
        for i in range(d + 1):
            cols.append((x1**(d-i)) * (x2**i))
    return np.column_stack(cols)

print("\n===== CASE 2: MULTIPLE REGRESSION (MANUAL) =====")
df = pd.read_csv("ml_fitbit.csv").dropna().drop_duplicates()
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

targets = ["Calories", "TotalMinutesAsleep"]

for target in targets:
    print(f"\n=========================================")
    print(f" TARGET: {target}")
    print(f"=========================================")
    
    # Pick top 2 most correlated features
    correlations = corr[target].drop(target).abs().sort_values(ascending=False)
    features = correlations.index[:2].tolist()
    print(f"Top 2 correlated features chosen: {features[0]} and {features[1]}")
    
    X = df[features].values
    y = df[target].values
    
    best_sse = float('inf')
    best_deg = 1
    best_theta = None
    
    for deg in range(1, 4): # Degrees 1 to 3 for bivariate
        start = time.time()
        
        # Build matrix and solve normal equation: theta = (X^T * X)^-1 * X^T * y
        X_poly = build_poly_features_2d(X, deg)
        
        # Using np.linalg.pinv (pseudo-inverse) is safer than inv for polynomial matrices 
        # to prevent singular matrix errors
        theta = np.linalg.pinv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)
        
        y_pred = X_poly.dot(theta)
        
        end = time.time()
        exec_time = end - start
        
        sse = np.sum((y - y_pred) ** 2)
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (sse / ss_tot)
        
        print(f"\nDegree {deg} Metrics:")
        print(f"SSE: {sse:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | Time: {exec_time:.6f}s")
        
        if sse < best_sse:
            best_sse = sse
            best_deg = deg
            best_theta = theta
            
    print(f"\n---> Best Polynomial Degree for {target}: {best_deg}")
    
    # 3D Plot of the best fit
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], y, color='red', alpha=0.5, label='Data')
    
    # Create a meshgrid for the surface
    x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
    x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    
    mesh_flat = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]
    mesh_poly = build_poly_features_2d(mesh_flat, best_deg)
    y_mesh_pred = mesh_poly.dot(best_theta).reshape(x1_mesh.shape)
    
    ax.plot_surface(x1_mesh, x2_mesh, y_mesh_pred, alpha=0.5, cmap='viridis')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(target)
    ax.set_title(f"Manual Multiple Reg: {target} (Deg {best_deg})")
    plt.savefig(f"case2_plot_{target}.png")
    plt.close()