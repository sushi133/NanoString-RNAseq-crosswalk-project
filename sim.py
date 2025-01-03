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

# Generate a correlated post-treatment data (using multivariate normal distribution)
# Define the covariance matrix between pre and post treatment values
cov_matrix = np.array([[1, correlation], [correlation, 1]])  # Correlation between pre and post-treatment data
mean_pre_post = np.zeros(2)  # Zero mean for both pre and post-treatment
pre_post_data = np.random.multivariate_normal(mean_pre_post, cov_matrix, size=(n_patients, n_genes))

# Separate pre-treatment and post-treatment data
platform1_post = platform1_pre * (1 + pre_post_data[:, 0])  # Scale post-treatment by correlation
platform2_post = platform2_pre * (1 + pre_post_data[:, 1])  # Scale post-treatment by correlation

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
