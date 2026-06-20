
# =========================================
# REGRESSION ASSIGNMENT - ALL 4 CASES
# =========================================
# Dataset assumed: ml_fitbit.csv
# Columns: SedentaryMinutes, TotalSteps, TotalMinutesAsleep, TotalTimeInBed, Calories

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import time

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("ml_fitbit.csv")

# Example targets
# You can switch between Calories or TotalMinutesAsleep

y = df["Calories"].values

# ==============================
# CASE 1: SIMPLE REGRESSION (MANUAL)
# ==============================
print("\n===== CASE 1: SIMPLE REGRESSION (MANUAL) =====")

x = df["TotalSteps"].values

# Polynomial degrees to test
max_degree = 8
best_sse = float('inf')
best_deg = 1

plt.figure()
plt.scatter(x, y)

for deg in range(1, max_degree + 1):
    coeffs = np.polyfit(x, y, deg)
    y_pred = np.polyval(coeffs, x)
    sse = np.sum((y - y_pred) ** 2)

    print(f"Degree {deg} SSE: {sse}")

    if sse < best_sse:
        best_sse = sse
        best_deg = deg
        best_curve = y_pred

plt.plot(x, best_curve)
plt.title(f"Best Degree: {best_deg}")
plt.savefig("plot_" + str(time.time()) + ".png")
plt.show()

print("Best degree:", best_deg)
print("Best SSE:", best_sse)

# ==============================
# CASE 2: MULTIPLE REGRESSION (MANUAL)
# ==============================
print("===== CASE 2: MULTIPLE REGRESSION (MANUAL) =====")

# Choose 2 predictors
X = df[["TotalSteps", "SedentaryMinutes"]].values

# Add bias column
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Normal equation
start = time.time()
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
end = time.time()

# Predictions
y_pred = X_b.dot(theta)

# Metrics
sse = np.sum((y - y_pred) ** 2)
mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y - y_pred))

print("SSE:", sse)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("Time:", end - start)

# 3D Plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], y)
ax.plot_trisurf(X[:,0], X[:,1], y_pred, alpha=0.5)

ax.set_xlabel("TotalSteps")
ax.set_ylabel("SedentaryMinutes")
ax.set_zlabel("Calories")
ax.set_title("Manual Multiple Regression")
plt.savefig("plot_" + str(time.time()) + ".png")
plt.show()

# ==============================
# CASE 3: SIMPLE REGRESSION (SKLEARN): SIMPLE REGRESSION (SKLEARN)
# ==============================
print("\n===== CASE 3: SIMPLE REGRESSION (SKLEARN) =====")

x = df[["TotalSteps"]].values

start = time.time()
model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)
end = time.time()

print("Intercept:", model.intercept_)
print("Slope:", model.coef_)
print("R2:", model.score(x, y))

print("SSE:", np.sum((y - y_pred) ** 2))
print("MSE:", metrics.mean_squared_error(y, y_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y, y_pred)))
print("MAE:", metrics.mean_absolute_error(y, y_pred))
print("Time:", end - start)

plt.figure()
plt.scatter(x, y)
plt.plot(x, y_pred)
plt.title("Simple Regression (sklearn)")
plt.savefig("plot_" + str(time.time()) + ".png")
plt.show()

# ==============================
# CASE 4: MULTIPLE REGRESSION (SKLEARN)
# ==============================
print("===== CASE 4: MULTIPLE REGRESSION (SKLEARN) =====")

X = df[["TotalSteps", "SedentaryMinutes"]].values

start = time.time()
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
end = time.time()

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("R2:", model.score(X, y))

print("SSE:", np.sum((y - y_pred) ** 2))
print("MSE:", metrics.mean_squared_error(y, y_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y, y_pred)))
print("MAE:", metrics.mean_absolute_error(y, y_pred))
print("Time:", end - start)

# 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], y)
ax.plot_trisurf(X[:,0], X[:,1], y_pred, alpha=0.5)

ax.set_xlabel("TotalSteps")
ax.set_ylabel("SedentaryMinutes")
ax.set_zlabel("Calories")
ax.set_title("Sklearn Multiple Regression")
plt.savefig("plot_" + str(time.time()) + ".png")
plt.show()

# ==============================
# OPTIONAL: POLYNOMIAL REGRESSION (SKLEARN): POLYNOMIAL REGRESSION (SKLEARN)
# ==============================
print("\n===== POLYNOMIAL REGRESSION (SKLEARN) =====")

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

print("R2:", model.score(X_poly, y))
print("SSE:", np.sum((y - y_pred) ** 2))

print("\n===== CORRELATION MATRIX =====")
corr = df.corr()
print(corr)

plt.figure()
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.show()