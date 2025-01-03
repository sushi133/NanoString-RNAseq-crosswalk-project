import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor  # Import Gradient Boosting

# Set random seed for reproducibility
np.random.seed(42)

# Simulate data
n_patients = 100  # Number of patients
n_genes = 1000    # Number of genes

# Define correlation structure between pre-treatment and post-treatment
correlation = 0.7  # High correlation between pre-treatment and post-treatment

# Generate random counts for Platform 1 (pre-treatment) from Poisson distribution
platform1_pre = np.random.poisson(lam=10, size=(n_patients, n_genes))

# Generate random counts for Platform 2 (pre-treatment) with linear transformation + noise
true_coefficients = np.random.normal(0, 0.5, n_genes)  # True relationship
noise = np.random.normal(0, 1, size=(n_patients, n_genes))
platform2_pre = np.dot(platform1_pre, np.diag(true_coefficients)) + noise

# Clip negative values for realistic counts
platform2_pre = np.clip(platform2_pre, 0, None)

# Generate correlated pre-treatment and post-treatment data for each gene
pre_treatment = np.column_stack([platform1_pre, platform2_pre])

# Initialize matrix for post-treatment data
post_treatment = np.zeros_like(pre_treatment)

# Create correlation matrix for each gene
for i in range(n_genes):
    # Generate correlated post-treatment data based on pre-treatment
    post_treatment[:, i] = pre_treatment[:, i] * (1 + np.random.normal(0, correlation, n_patients))

# Split pre-treatment and post-treatment data
platform1_post = post_treatment[:, :n_genes]  # First half for platform1 post-treatment
platform2_post = post_treatment[:, n_genes:]  # Second half for platform2 post-treatment

# Clip negative values for post-treatment counts
platform1_post = np.clip(platform1_post, 0, None)
platform2_post = np.clip(platform2_post, 0, None)

# Convert to DataFrames
df_platform1_pre = pd.DataFrame(platform1_pre, columns=[f"gene_{i+1}" for i in range(n_genes)])
df_platform2_pre = pd.DataFrame(platform2_pre, columns=[f"gene_{i+1}" for i in range(n_genes)])
df_platform1_post = pd.DataFrame(platform1_post, columns=[f"gene_{i+1}" for i in range(n_genes)])
df_platform2_post = pd.DataFrame(platform2_post, columns=[f"gene_{i+1}" for i in range(n_genes)])

# Combine pre-treatment and post-treatment data into one DataFrame
df_platform1 = pd.concat([df_platform1_pre, df_platform1_post], axis=1, keys=["pre_treatment", "post_treatment"])
df_platform2 = pd.concat([df_platform2_pre, df_platform2_post], axis=1, keys=["pre_treatment", "post_treatment"])

# Combine both pre- and post-treatment data for training and testing
X = pd.concat([df_platform1["pre_treatment"], df_platform1["post_treatment"]], axis=1)
y = pd.concat([df_platform2["pre_treatment"], df_platform2["post_treatment"]], axis=1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Lasso Regression ---
lasso = Lasso(alpha=0.1)  # Adjust alpha for stronger/weaker regularization
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# --- Gradient Boosting (XGBoost) ---
xgb_model = XGBRegressor(
    n_estimators=100,    # Number of trees
    max_depth=6,         # Maximum depth of a tree
    learning_rate=0.1,   # Step size shrinkage
    random_state=42      # For reproducibility
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# --- Performance Comparison ---
print("Performance Metrics:")
print(f"Lasso Regression - Mean Squared Error (MSE): {mse_lasso:.2f}, R-squared: {r2_lasso:.2f}")
print(f"Gradient Boosting - Mean Squared Error (MSE): {mse_xgb:.2f}, R-squared: {r2_xgb:.2f}")

# Display some true vs predicted counts for Lasso Regression
comparison_lasso = pd.DataFrame({
    "True Counts": y_test.iloc[0].values,
    "Predicted Counts (Lasso)": y_pred_lasso[0]
})
print("\nComparison of True vs Predicted Counts (Lasso):")
print(comparison_lasso.head(10))

# Display some true vs predicted counts for Gradient Boosting
comparison_xgb = pd.DataFrame({
    "True Counts": y_test.iloc[0].values,
    "Predicted Counts (XGBoost)": y_pred_xgb[0]
})
print("\nComparison of True vs Predicted Counts (XGBoost):")
print(comparison_xgb.head(10))
