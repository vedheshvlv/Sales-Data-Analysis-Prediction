import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("sales.csv")

print("Dataset Preview:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Data Cleaning
# -----------------------------
df = df.dropna()

# -----------------------------
# Exploratory Data Analysis
# -----------------------------

# Graph 1 - Distribution of Sales
plt.figure(figsize=(8,5))
sns.histplot(df["OutletSales"], bins=30, kde=True)
plt.title("Distribution of Outlet Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()

# Graph 2 - Outlet Type vs Sales
plt.figure(figsize=(8,5))
df.groupby("OutletType")["OutletSales"].mean().plot(kind="bar")
plt.title("Average Sales by Outlet Type")
plt.xlabel("Outlet Type")
plt.ylabel("Average Sales")
plt.xticks(rotation=45)
plt.show()

# Graph 3 - Fat Content vs Sales
plt.figure(figsize=(8,5))
df.groupby("FatContent")["OutletSales"].mean().plot(kind="bar")
plt.title("Fat Content vs Average Sales")
plt.xlabel("Fat Content")
plt.ylabel("Average Sales")
plt.xticks(rotation=45)
plt.show()

# Graph 4 - Correlation Heatmap
plt.figure(figsize=(10,6))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# Feature Selection
# -----------------------------
features = [
    "Weight",
    "FatContent",
    "ProductVisibility",
    "OutletType",
    "OutletSize",
    "LocationType",
    "MRP",
    "EstablishmentYear"
]

X = df[features]
y = df["OutletSales"]

# -----------------------------
# Convert categorical columns
# -----------------------------
X = pd.get_dummies(X, drop_first=True)

print("\nProcessed Features:")
print(X.head())

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# LINEAR REGRESSION
# =====================================================

lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("\n===== Linear Regression =====")
print("MAE :", round(lr_mae, 2))
print("RMSE:", round(lr_rmse, 2))
print("R2 Score:", round(lr_r2, 2))

# =====================================================
# RANDOM FOREST
# =====================================================

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n===== Random Forest =====")
print("MAE :", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("R2 Score:", round(rf_r2, 2))

# =====================================================
# GRADIENT BOOSTING
# =====================================================

gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))
gb_r2 = r2_score(y_test, gb_pred)

print("\n===== Gradient Boosting =====")
print("MAE :", round(gb_mae, 2))
print("RMSE:", round(gb_rmse, 2))
print("R2 Score:", round(gb_r2, 2))

# =====================================================
# XGBOOST
# =====================================================

xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print("\n===== XGBoost =====")
print("MAE :", round(xgb_mae, 2))
print("RMSE:", round(xgb_rmse, 2))
print("R2 Score:", round(xgb_r2, 2))

# =====================================================
# MODEL COMPARISON GRAPH
# =====================================================

models = [
    "Linear Regression",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost"
]

mae_values = [
    lr_mae,
    rf_mae,
    gb_mae,
    xgb_mae
]

rmse_values = [
    lr_rmse,
    rf_rmse,
    gb_rmse,
    xgb_rmse
]

# MAE Comparison
plt.figure(figsize=(10,5))
plt.bar(models, mae_values)
plt.title("Model Comparison using MAE")
plt.xlabel("Models")
plt.ylabel("MAE")
plt.xticks(rotation=10)
plt.show()

# RMSE Comparison
plt.figure(figsize=(10,5))
plt.bar(models, rmse_values)
plt.title("Model Comparison using RMSE")
plt.xlabel("Models")
plt.ylabel("RMSE")
plt.xticks(rotation=10)
plt.show()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = rf.feature_importances_

feature_names = X.columns

plt.figure(figsize=(12,6))
plt.bar(feature_names, importance)
plt.xticks(rotation=90)
plt.title("Feature Importance - Random Forest")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()

# =====================================================
# FINAL PREDICTION GRAPH
# =====================================================

plt.figure(figsize=(12,6))

plt.plot(y_test.values[:100], label="Actual Sales")
plt.plot(rf_pred[:100], label="Random Forest")
plt.plot(xgb_pred[:100], label="XGBoost")

plt.title("Actual vs Predicted Sales")
plt.xlabel("Samples")
plt.ylabel("Sales")
plt.legend()
plt.show()
