# =========================================
# CASE 4: MULTIPLE REGRESSION (SKLEARN)
# =========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import time

print("\n===== CASE 4: MULTIPLE REGRESSION (SKLEARN) =====")
df = pd.read_csv("ml_fitbit.csv").dropna().drop_duplicates()
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

targets = ["Calories", "TotalMinutesAsleep"]

for target in targets:
    print(f"\n=========================================")
    print(f" TARGET: {target}")
    print(f"=========================================")
    
    correlations = corr[target].drop(target).abs().sort_values(ascending=False)
    features = correlations.index[:2].tolist()
    print(f"Top 2 correlated features chosen: {features[0]} and {features[1]}")
    
    X = df[features].values
    y = df[target].values
    
    best_sse = float('inf')
    best_deg = 1
    best_model = None
    best_poly = None
    
    for deg in range(1, 4): # Degrees 1 to 3
        start = time.time()
        
        poly = PolynomialFeatures(degree=deg)
        X_poly = poly.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)
        
        end = time.time()
        exec_time = end - start
        
        mse = metrics.mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(y, y_pred)
        r2 = metrics.r2_score(y, y_pred)
        sse = mse * len(y)
        
        print(f"\nDegree {deg} Metrics:")
        print(f"SSE: {sse:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | Time: {exec_time:.6f}s")
        
        if sse < best_sse:
            best_sse = sse
            best_deg = deg
            best_model = model
            best_poly = poly
            
    print(f"\n---> Best Polynomial Degree for {target}: {best_deg}")
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], y, color='red', alpha=0.5)
    
    x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
    x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    
    mesh_flat = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]
    mesh_poly = best_poly.transform(mesh_flat)
    y_mesh_pred = best_model.predict(mesh_poly).reshape(x1_mesh.shape)
    
    ax.plot_surface(x1_mesh, x2_mesh, y_mesh_pred, alpha=0.5, cmap='plasma')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(target)
    ax.set_title(f"Sklearn Multiple Reg: {target} (Deg {best_deg})")
    plt.savefig(f"case4_plot_{target}.png")
    plt.close()