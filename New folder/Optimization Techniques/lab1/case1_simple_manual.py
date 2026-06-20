# =========================================
# CASE 1: SIMPLE REGRESSION (MANUAL)
# =========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

print("\n===== EXPLORATORY DATA ANALYSIS (EDA) =====")
df = pd.read_csv("ml_fitbit.csv")

# Data Cleaning
print(f"Original shape: {df.shape}")
df = df.dropna()
df = df.drop_duplicates()
print(f"Shape after dropping NaNs and duplicates: {df.shape}\n")

print("--- DataFrame Info ---")
df.info()
print("\n--- DataFrame Describe ---")
print(df.describe())

print("\n===== CORRELATION MATRIX =====")
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
print(corr)

# Save correlation matrix plot
plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right')
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()

targets = ["Calories", "TotalMinutesAsleep"]

for target in targets:
    print(f"\n=========================================")
    print(f" TARGET: {target}")
    print(f"=========================================")
    
    # Find the single most correlated feature (absolute value, excluding the target itself)
    correlations = corr[target].drop(target).abs()
    best_feature = correlations.idxmax()
    print(f"Most correlated feature chosen: {best_feature} (r = {corr[target][best_feature]:.4f})")
    
    x = df[best_feature].values
    y = df[target].values
    
    best_sse = float('inf')
    best_deg = 1
    best_metrics = {}
    best_coeffs = None
    
    for deg in range(1, 9): # Degrees 1 to 8
        start = time.time()
        
        # Manual fit
        coeffs = np.polyfit(x, y, deg)
        y_pred = np.polyval(coeffs, x)
        
        end = time.time()
        exec_time = end - start
        
        # Calculate Metrics Manually
        sse = np.sum((y - y_pred) ** 2)
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        
        # Manual R^2
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (sse / ss_tot)
        
        print(f"\nDegree {deg} Metrics:")
        print(f"SSE: {sse:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | Time: {exec_time:.6f}s")
        
        if sse < best_sse:
            best_sse = sse
            best_deg = deg
            best_coeffs = coeffs
            best_metrics = {'SSE': sse, 'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
            
    print(f"\n---> Best Polynomial Degree for {target}: {best_deg}")
    
    # Plotting best curve
    plt.figure()
    plt.scatter(x, y, color='red', alpha=0.5, label='Data points')
    
    # Generate smooth line for polynomial curve
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = np.polyval(best_coeffs, x_line)
    
    plt.plot(x_line, y_line, color='blue', label=f'Degree {best_deg} Fit')
    plt.xlabel(best_feature)
    plt.ylabel(target)
    plt.title(f"Manual Simple Reg: {target} vs {best_feature} (Deg {best_deg})")
    plt.legend()
    plt.savefig(f"case1_plot_{target}.png")
    plt.close()