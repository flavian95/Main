# =========================================
# CASE 3: SIMPLE REGRESSION (SKLEARN)
# =========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import time

print("\n===== CASE 3: SIMPLE REGRESSION (SKLEARN) =====")
df = pd.read_csv("ml_fitbit.csv").dropna().drop_duplicates()
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

targets = ["Calories", "TotalMinutesAsleep"]

for target in targets:
    print(f"\n=========================================")
    print(f" TARGET: {target}")
    print(f"=========================================")
    
    correlations = corr[target].drop(target).abs()
    best_feature = correlations.idxmax()
    print(f"Most correlated feature chosen: {best_feature}")
    
    # Sklearn requires 2D arrays for X
    x = df[[best_feature]].values 
    y = df[target].values
    
    best_sse = float('inf')
    best_deg = 1
    best_model = None
    best_poly = None
    
    for deg in range(1, 9): # Degrees 1 to 8
        start = time.time()
        
        poly = PolynomialFeatures(degree=deg)
        x_poly = poly.fit_transform(x)
        
        model = LinearRegression()
        model.fit(x_poly, y)
        y_pred = model.predict(x_poly)
        
        end = time.time()
        exec_time = end - start
        
        # Using sklearn metrics
        mse = metrics.mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(y, y_pred)
        r2 = metrics.r2_score(y, y_pred)
        sse = mse * len(y) # Sklearn doesn't have direct SSE
        
        print(f"\nDegree {deg} Metrics:")
        print(f"SSE: {sse:.4f} | MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f} | Time: {exec_time:.6f}s")
        
        if sse < best_sse:
            best_sse = sse
            best_deg = deg
            best_model = model
            best_poly = poly
            
    print(f"\n---> Best Polynomial Degree for {target}: {best_deg}")
    
    plt.figure()
    plt.scatter(x, y, color='red', alpha=0.5, label='Data points')
    
    # Sort values for smooth plotting
    x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    x_line_poly = best_poly.transform(x_line)
    y_line = best_model.predict(x_line_poly)
    
    plt.plot(x_line, y_line, color='blue', label=f'Degree {best_deg} Sklearn Fit')
    plt.xlabel(best_feature)
    plt.ylabel(target)
    plt.title(f"Sklearn Simple Reg: {target} vs {best_feature}")
    plt.legend()
    plt.savefig(f"case3_plot_{target}.png")
    plt.close()