import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Simulate data
n_patients = 100  # Number of patients
n_genes = 1000    # Number of genes
correlation = 0.7  # Correlation between pre- and post-treatment within subjects

# Create a random correlation matrix between genes (e.g., 10 specific genes with higher correlation)
correlated_gene_indices = np.random.choice(n_genes, size=10, replace=False)  # Select 10 genes to correlate
corr_matrix_genes = np.eye(n_genes)  # Identity matrix for no correlation
for i in range(len(correlated_gene_indices)):
    for j in range(i + 1, len(correlated_gene_indices)):
        gene1 = correlated_gene_indices[i]
        gene2 = correlated_gene_indices[j]
        corr_matrix_genes[gene1, gene2] = correlation
        corr_matrix_genes[gene2, gene1] = correlation

# Generate latent correlated data for the genes
latent_data = np.random.multivariate_normal(
    mean=np.zeros(n_genes), cov=corr_matrix_genes, size=n_patients
)

# Transform latent data into Poisson-distributed data
platform1_pre = np.exp(latent_data)  # Exponential to ensure positive values
platform1_pre = np.random.poisson(lam=platform1_pre)  # Poisson transformation

# Clip negative values for realistic counts (if any)
platform1_pre = np.clip(platform1_pre, 0, None)

# Generate Platform 2 (pre-treatment) data with linear transformation + noise
true_coefficients = np.random.normal(0, 0.5, n_genes)
noise = np.random.normal(0, 1, size=(n_patients, n_genes))
platform2_pre = np.dot(platform1_pre, np.diag(true_coefficients)) + noise
platform2_pre = np.clip(platform2_pre, 0, None)  # Ensure non-negative values
platform2_pre = np.round(platform2_pre).astype(int)  # Convert to integers

# Combine pre-treatment data for both platforms
pre_treatment_combined = np.hstack((platform1_pre, platform2_pre))

# Generate correlated data of shape (n_patients, 2*n_genes)
correlated_data = np.random.normal(0, correlation, size=(n_patients, 2*n_genes))  # Adjust correlation strength if necessary

# Apply correlation to generate post-treatment
post_treatment = pre_treatment_combined * (1 + correlated_data)  # Apply correlation
post_treatment = np.clip(post_treatment, 0, None)  # Ensure non-negative values
post_treatment = np.round(post_treatment).astype(int)  # Convert to integers

# Split post-treatment data for platform1 and platform2
platform1_post = post_treatment[:, :n_genes]
platform2_post = post_treatment[:, n_genes:]

# Convert pre- and post-treatment data to DataFrames
def create_dataframe(platform_pre, platform_post, n_genes):
    df_pre = pd.DataFrame(platform_pre, columns=[f"gene_{i+1}" for i in range(n_genes)])
    df_post = pd.DataFrame(platform_post, columns=[f"gene_{i+1}" for i in range(n_genes)])
    return pd.concat([df_pre, df_post], axis=1, keys=["pre_treatment", "post_treatment"])

df_platform1 = create_dataframe(platform1_pre, platform1_post, n_genes)
df_platform2 = create_dataframe(platform2_pre, platform2_post, n_genes)

# Combine both pre- and post-treatment data for training and testing
X = pd.concat([df_platform1["pre_treatment"], df_platform1["post_treatment"]], axis=1)
y = pd.concat([df_platform2["pre_treatment"], df_platform2["post_treatment"]], axis=1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Lasso Regression ---
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)
y_pred_lasso = np.clip(y_pred_lasso, 0, None)  # Ensure non-negative values
y_pred_lasso = np.round(y_pred_lasso).astype(int)  # Convert to integers
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
ks_lasso = ks_2samp(y_test.to_numpy().ravel(), y_pred_lasso.ravel()).statistic

# --- Gradient Boosting (XGBoost) ---
xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train.to_numpy(), y_train.to_numpy())
y_pred_xgb = xgb_model.predict(X_test.to_numpy())
y_pred_xgb = np.clip(y_pred_xgb, 0, None)  # Ensure non-negative values
y_pred_xgb = np.round(y_pred_xgb).astype(int)  # Convert to integers
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
ks_xgb = ks_2samp(y_test.to_numpy().ravel(), y_pred_xgb.ravel()).statistic

# --- Performance Comparison ---
print("Performance Metrics:")
print(f"Lasso Regression - Mean Squared Error (MSE): {mse_lasso:.2f}, R-squared: {r2_lasso:.2f}, KS Distance: {ks_lasso:.2f}")
print(f"Gradient Boosting - Mean Squared Error (MSE): {mse_xgb:.2f}, R-squared: {r2_xgb:.2f}, KS Distance: {ks_xgb:.2f}")

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

# Create a figure with two subplots (side-by-side)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Lasso Residual Plot (Left)
axes[0].scatter(
    y_test.to_numpy().ravel(),
    residuals_lasso,
    alpha=0.5,
    color="blue",
    label="Lasso Residuals",
)
axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[0].set_xlabel("True Counts")
axes[0].set_ylabel("Residuals (True - Predicted)")
axes[0].set_title("Lasso Regression Residuals")
axes[0].legend()
axes[0].grid(True)

# XGBoost Residual Plot (Right)
axes[1].scatter(
    y_test.to_numpy().ravel(),
    residuals_xgb,
    alpha=0.5,
    color="green",
    label="XGBoost Residuals",
)
axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("True Counts")
axes[1].set_title("XGBoost Residuals")
axes[1].legend()
axes[1].grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
