import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Read the file
rnaseq = pd.read_csv("RNAseq_norm_nanofilt_Adzib_1.22.25.txt", sep=" ")
nanostring = pd.read_csv("Nano_norm_James_1.22.25.txt", sep=" ")

# Print first 5 lines
rnaseq.head()
nanostring.head()

# Melt data to long format
rnaseq_long = rnaseq.melt(id_vars=["Geneid"], var_name="Patient_Time", value_name="RNAseq_Exp")
nanostring_long = nanostring.melt(id_vars=["Geneid"], var_name="Patient_Time", value_name="NanoString_Exp")

# Subset rnaseq_long to the first 100 genes for testing
# rnaseq_long = rnaseq_long[rnaseq_long['Geneid'].isin(rnaseq_long['Geneid'].unique()[:1000])]
# Subset nanostring_long to the first 10 genes for testing
# nanostring_long = nanostring_long[nanostring_long['Geneid'].isin(nanostring_long['Geneid'].unique()[:1])]

# Extract Patient and Time info from Patient_Time column
rnaseq_long[['Patient', 'Time']] = rnaseq_long['Patient_Time'].str.split('_', expand=True)
nanostring_long[['Patient', 'Time']] = nanostring_long['Patient_Time'].str.split('_', expand=True)

# Loop through each NanoString gene and predict using all RNA-seq genes
genes_nano = nanostring_long['Geneid'].unique()

lasso_models = {}

# Initialize empty list to store results for evaluation
results = []

# Initialize empty list to store residuals for plotting later
all_residuals = []
all_true_values = []

# Loop through each NanoString gene and predict using all RNA-seq genes
for gene in genes_nano:
    # Get the data for the specific NanoString gene
    df_gene_nano = nanostring_long[nanostring_long['Geneid'] == gene]
    
    # Use all RNA-seq genes as features
    df_gene_rnaseq = rnaseq_long[rnaseq_long['Patient'].isin(df_gene_nano['Patient']) & 
                                  rnaseq_long['Time'].isin(df_gene_nano['Time'])]
    
    # Pivot RNA-seq data to get gene expression values as features
    rnaseq_pivot = df_gene_rnaseq.pivot(index=['Patient', 'Time'], columns='Geneid', values='RNAseq_Exp').reset_index()
    
    # Merge RNA-seq data with NanoString data on Patient and Time
    merged_data = pd.merge(df_gene_nano, rnaseq_pivot, on=['Patient', 'Time'], how='inner')
    
    # Encode categorical variables
    merged_data['Patient'] = merged_data['Patient'].astype('category').cat.codes  # Encode patient IDs
    merged_data['Time'] = merged_data['Time'].map({'Pre': 0, 'Post': 1})  # Encode time as binary
    
    # Prepare feature matrix and target vector
    X = merged_data.drop(columns=['Geneid', 'NanoString_Exp', 'Patient_Time'])  # Drop columns that are not features
    X = sm.add_constant(X)  # Add intercept
    y = merged_data['NanoString_Exp']
    
    # Feature selection: Fit Lasso regression for feature selection
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
    lasso.fit(X, y)
    
    # Feature selection: Get non-zero features
    non_zero_features = np.where(lasso.coef_ != 0)[0]
    
    # Filter the feature matrix based on selected features
    X_selected = X.iloc[:, non_zero_features]
    
    # Add the constant back (intercept) since it was dropped in feature selection
    X_selected = sm.add_constant(X_selected)
    
    # Handle the case where only the intercept is selected
    if X_selected.shape[1] == 1:  # Only the intercept is selected
        formula = "NanoString_Exp ~ 1"  # No explanatory variables, just intercept
    else:
        predictors = ' + '.join([f'Q("{col}")' for col in X_selected.columns[1:]])
        formula = f"NanoString_Exp ~ {predictors}"
    
    # For the random effect, we specify '(1|Patient)' as the random intercept for each patient
    mixed_model = smf.mixedlm(formula, merged_data, groups=merged_data['Patient'])
    mixed_results = mixed_model.fit()
    
    # Make predictions using the mixed model
    merged_data['predicted'] = mixed_results.predict(merged_data)
    
    # Calculate residuals for mixed model (Predicted - True) / True
    residuals_mixed = (merged_data['predicted'] - y) / y
    
    # Append residuals and true values for later plotting
    all_residuals.extend(residuals_mixed)
    all_true_values.extend(y)
    
    # Evaluate the performance of the mixed model predictions
    mse = mean_squared_error(y, merged_data['predicted'])
    r2 = r2_score(y, merged_data['predicted'])
    
    # Append results for evaluation
    results.append({
        'Gene': gene,
        'MSE': mse,
        'R2': r2
    })
    
    # Print evaluation results
    print(f"Gene {gene}: Mean Squared Error (MSE) = {mse}")
    print(f"Gene {gene}: R2 Score = {r2}")
    
    # Print the predicted counts vs true counts
    print(f"Gene {gene}: Predicted vs True counts")
    comparison = pd.DataFrame({'True': y, 'Predicted': merged_data['predicted']})
    print(comparison.head())  # Print the first few rows of comparison

# Create a figure for Mixed Model Residual Plot after the loop
plt.figure(figsize=(8, 6))
plt.scatter(all_true_values, all_residuals, alpha=0.5, color="green", label="Mixed Model Residuals")
plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
plt.xscale("log")  # Set x-axis to log scale
plt.xlabel("True Counts (Log Scale)")
plt.ylabel("Fold Change: (Predicted - True) / True")
plt.title("Mixed Model Residuals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
