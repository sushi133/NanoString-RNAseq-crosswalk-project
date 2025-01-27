import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Read data
rnaseq = pd.read_csv("RNAseq_norm_nanofilt_Adzib_1.22.25.txt", sep=" ")
nanostring = pd.read_csv("Nano_norm_James_1.22.25.txt", sep=" ")

# Melt data to long format
rnaseq_long = rnaseq.melt(id_vars=["Geneid"], var_name="Patient_Time", value_name="RNAseq_Exp")
nanostring_long = nanostring.melt(id_vars=["Geneid"], var_name="Patient_Time", value_name="NanoString_Exp")

# Extract Patient and Time info
rnaseq_long[['Patient', 'Time']] = rnaseq_long['Patient_Time'].str.split('_', expand=True)
nanostring_long[['Patient', 'Time']] = nanostring_long['Patient_Time'].str.split('_', expand=True)

# Initialize lists for residuals
all_residuals, all_true_values = [], []

# Loop through each NanoString gene
for gene in nanostring_long['Geneid'].unique():
    df_gene_nano = nanostring_long[nanostring_long['Geneid'] == gene]

    # Pivot RNA-seq data for use as predictors
    rnaseq_pivot = rnaseq_long.pivot(index=['Patient', 'Time'], columns='Geneid', values='RNAseq_Exp').reset_index()
    merged_data = pd.merge(df_gene_nano, rnaseq_pivot, on=['Patient', 'Time'], how='inner')

    # Encode categorical variables
    merged_data = merged_data.assign(
        Patient=merged_data['Patient'].astype('category').cat.codes,
        Time=merged_data['Time'].map({'Pre': 0, 'Post': 1})
    )

    # Prepare feature matrix and target
    X = merged_data.drop(columns=['Geneid', 'NanoString_Exp', 'Patient_Time'])
    y = merged_data['NanoString_Exp']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model and compute residuals
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    residuals = (model.predict(X_test) - y_test) / y_test

    all_residuals.extend(residuals)
    all_true_values.extend(y_test)

# Create DataFrame for residuals
residuals_df = pd.DataFrame({'True_Counts': all_true_values, 'Residuals': all_residuals})
# Save to CSV file
residuals_df.to_csv("GB_model_residuals.csv", index=False)
# Load the saved residuals dataset
residuals_df = pd.read_csv("GB_model_residuals.csv")

# Create a figure for Mixed Model Residual Plot
plt.figure(figsize=(8, 6))
plt.scatter(residuals_df['True_Counts'], residuals_df['Residuals'], alpha=0.5, color="green", label="Gradient Boosting Model Performance")
plt.axhline(1, color="red", linestyle="--", linewidth=1.5)
plt.axhline(-1, color="red", linestyle="--", linewidth=1.5)
plt.xscale("log")  # Set x-axis to log scale
# plt.ylim(-10, 50)  # Set y-axis limits
plt.xlabel("True Counts (Log Scale)")
plt.ylabel("Fold Change: (Predicted - True) / True")
plt.title("Gradient Boosting Model Performance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Compute acceptable range based on a twofold change
acceptable = (residuals_df['Residuals'] >= -1) & (residuals_df['Residuals'] <= 1)
# Calculate the percentage of predictions within the acceptable range
acceptable_percentage = acceptable.mean() * 100  # Convert to percentage
# Print the result
print(f"Percentage of predicted values within a twofold change: {acceptable_percentage:.2f}%")