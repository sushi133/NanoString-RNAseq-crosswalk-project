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

# Initialize lists 
all_predict_values, all_true_values, all_genes = [], [], []

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

    # Store predict values with true values and gene names
    all_predict_values.extend(model.predict(X_test))
    all_true_values.extend(y_test)
    all_genes.extend([gene] * len(y_test))  # Store the corresponding gene 

# Create DataFrame for residuals
df = pd.DataFrame({'Geneid': all_genes, 'True_Counts': all_true_values, 'Predict_Counts': all_predict_values})
# Save to CSV file
df.to_csv("GB_model.csv", index=False)
# Load the saved residuals dataset
df = pd.read_csv("GB_model.csv")

# Calculate the absolute fold change (max of True_Counts / Predict_Counts or Predict_Counts / True_Counts)
df['Abs_Fold_Change'] = np.maximum(df['True_Counts'] / df['Predict_Counts'], df['Predict_Counts'] / df['True_Counts'])
# Flag genes with an absolute fold change greater than 10
df['Bad_Gene'] = df['Abs_Fold_Change'] > 10
# Filter bad genes
bad_genes = df[df['Bad_Gene']]
# Save or display bad genes
bad_genes.to_csv("bad_genes.csv", index=False)
# Optionally, print out the bad genes
print(bad_genes)

# Plot the fold change against the true counts
plt.figure(figsize=(8, 6))
plt.scatter(df['True_Counts'], df['Abs_Fold_Change'], alpha=0.5, color="green", label="Gradient Boosting Model Performance")
plt.axhline(10, color="red", linestyle="--", linewidth=1.5)
# Set x-axis to log scale
plt.xscale("log")
plt.xlabel("True Counts")
plt.ylabel("Fold Change")
plt.title("Gradient Boosting Model Performance")
plt.legend()
plt.grid(True)
plt.tight_layout()
# Show the plot
plt.show()

# Compute acceptable range based on a 10-fold change
acceptable = (df['Abs_Fold_Change'] <= 10) 
# Calculate the percentage of predictions within the acceptable range
acceptable_percentage = acceptable.mean() * 100  # Convert to percentage
# Print the result
print(f"Percentage of predicted values within a 10-fold change: {acceptable_percentage:.2f}%")