import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("sales.csv")

print("Dataset Preview:")
print(df.head())

# -----------------------------
# 📊 EDA GRAPHS (ADD HERE)
# -----------------------------

# Graph 1 — Distribution of Sales
plt.figure()
sns.histplot(df["OutletSales"], bins=30, kde=True)
plt.title("Distribution of Outlet Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

# Graph 2 — Average Sales by Outlet Type
plt.figure()
df.groupby("OutletType")["OutletSales"].mean().plot(kind="bar")
plt.title("Average Sales by Outlet Type")
plt.xlabel("Outlet Type")
plt.ylabel("Average Sales")
plt.show()

# Graph 3 — Correlation Heatmap
plt.figure()
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# Data Cleaning
# -----------------------------
df = df.dropna()

sales_column = "OutletSales"

# Create numeric feature using index
df["Index"] = np.arange(len(df))

X = df[["Index"]]
y = df[sales_column]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# Linear Regression
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nLinear Regression Performance:")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))

# -----------------------------
# Random Forest Model
# -----------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("\nRandom Forest Performance:")
print("MAE :", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))

# -----------------------------
# Final Prediction Graph
# -----------------------------
plt.figure()
plt.plot(y_test.values, label="Actual Sales")
plt.plot(y_pred, label="Linear Regression")
plt.plot(rf_pred, label="Random Forest")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Samples")
plt.ylabel("Sales")
plt.legend()
plt.show()
