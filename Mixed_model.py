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

# Initialize lists for fold change
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
    y_pred = model.predict(X_test_selected)  # Use test data for predictions
    residuals = (y_pred - y_test) / y_test  # Calculate residuals on test set
    all_residuals.extend(residuals)
    all_true_values.extend(y_test)

# Create a DataFrame from the lists
residuals_df = pd.DataFrame({'True_Counts': all_true_values, 'Residuals': all_residuals})
# Save to CSV file
residuals_df.to_csv("mixed_model_residuals.csv", index=False)
# Load the saved residuals dataset
residuals_df = pd.read_csv("mixed_model_residuals.csv")

# Create a figure for Mixed Model Residual Plot
plt.figure(figsize=(8, 6))
plt.scatter(residuals_df['True_Counts'], residuals_df['Residuals'], alpha=0.5, color="green", label="Mixed-Effects Model Performance")
plt.axhline(1, color="red", linestyle="--", linewidth=1.5)
plt.axhline(-1, color="red", linestyle="--", linewidth=1.5)
plt.xscale("log")  # Set x-axis to log scale
# plt.ylim(-10, 50)  # Set y-axis limits
plt.xlabel("True Counts (Log Scale)")
plt.ylabel("Fold Change: (Predicted - True) / True")
plt.title("Mixed-Effects Model Performance")
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