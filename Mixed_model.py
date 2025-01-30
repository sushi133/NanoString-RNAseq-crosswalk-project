import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Load data
rnaseq = pd.read_csv("RNAseq_norm_nanofilt_Adzib_1.22.25.txt", sep=" ")
nanostring = pd.read_csv("Nano_norm_James_1.22.25.txt", sep=" ")

# Melt data to long format
rnaseq_long = rnaseq.melt(id_vars=["Geneid"], var_name="Patient_Time", value_name="RNAseq_Exp")
nanostring_long = nanostring.melt(id_vars=["Geneid"], var_name="Patient_Time", value_name="NanoString_Exp")

# Extract Patient and Time info
rnaseq_long[['Patient', 'Time']] = rnaseq_long['Patient_Time'].str.split('_', expand=True)
nanostring_long[['Patient', 'Time']] = nanostring_long['Patient_Time'].str.split('_', expand=True)

# Initialize lists
all_predict_values, all_true_values, all_genes, all_fold_changes, all_log2_fold_changes = [], [], [], [], []

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

    # Feature selection with Lasso (only on training data)
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000).fit(X_train, y_train)
    selected_features = np.where(lasso.coef_ != 0)[0]

    # Select features and add constant
    X_train_selected = sm.add_constant(X_train.iloc[:, selected_features])
    X_test_selected = sm.add_constant(X_test.iloc[:, selected_features])

    # Fit mixed model (only using training data)
    formula = "NanoString_Exp ~ " + " + ".join([f'Q("{col}")' for col in X_train_selected.columns[1:]]) if selected_features.size else "NanoString_Exp ~ 1"
    train_data = merged_data.loc[X_train.index]  # Use only training data
    model = smf.mixedlm(formula, train_data, groups=train_data['Patient']).fit()

    # Predictions on test data
    predicted_values = model.predict(X_test_selected)  # Use test data for predictions
    # Calculate fold change with conditional behavior within the loop
    fold_change = np.where(
        predicted_values >= y_test,
        predicted_values / y_test,
        - (y_test / predicted_values)
    )

    # Compute log2 fold change with pseudo-count to avoid log(0)
    log2_fold_change = np.log2((predicted_values + 1) / (y_test + 1))

    # Add results to lists
    all_predict_values.extend(predicted_values)
    all_true_values.extend(y_test)
    all_genes.extend([gene] * len(y_test))  # Store the corresponding gene
    all_fold_changes.extend(fold_change)
    all_log2_fold_changes.extend(log2_fold_change)

# Create DataFrame for results
final_df = pd.DataFrame({
    'Geneid': all_genes,
    'True_Counts': all_true_values,
    'Predict_Counts': all_predict_values,
    'Fold_Change': all_fold_changes,
    'Log2_Fold_Change': all_log2_fold_changes
})

# Save to CSV file
final_df.to_csv("mixed_model.csv", index=False)
# Load the saved dataset
final_df = pd.read_csv("mixed_model.csv")

# Plot the fold change against the true counts
plt.figure(figsize=(8, 6))
plt.scatter(final_df['True_Counts'], final_df['Fold_Change'], alpha=0.5, color="green", label="Mixed Effect Model Performance")
# plt.scatter(final_df['True_Counts'], final_df['Fold_Change'], alpha=0.5, color="green", label="Gradient Boosting Model Performance")
plt.xscale("log")
plt.xlabel("True Counts")
# plt.ylabel("Signed Fold Change")
plt.ylabel("Signed Fold Change")
plt.title("Mixed Effect Model Performance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()