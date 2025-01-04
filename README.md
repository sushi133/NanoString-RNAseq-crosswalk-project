# Genomic Count Data Simulation and Model Comparison

This script simulates genomic count data with pre-treatment and post-treatment measurements on two platforms. It compares the performance of Lasso regression and XGBoost models in predicting data from a new platform (Platform 2) based on data from an original platform (Platform 1). The simulation includes various correlations: between pre- and post-treatment values, between certain genes, and between platforms.

# Key Steps:

## Simulate Data:

Platform 1 Pre-treatment Data: The pre-treatment data for Platform 1 is generated using the Poisson distribution based on a correlated set of genes. A correlation matrix is created for 10 genes (out of 1000 total) to reflect higher correlations between those specific genes.

Platform 2 Pre-treatment Data: Generated using a linear transformation of Platform 1 data plus noise, introducing correlation between platforms.

Post-treatment Data: The post-treatment data is generated by applying a correlation structure between pre- and post-treatment values, which reflects the correlation between both platforms and the correlation among genes in the dataset.

## Prepare and Split Data for Training and Testing:

The data for both platforms is combined and then split into training and testing sets.

The features are standardized using StandardScaler to ensure the models perform optimally.

## Fit Lasso Regression and XGBoost Models:

Lasso Regression: A linear regression model with L1 regularization is used to predict Platform 2 data.

XGBoost: A gradient boosting model is used to predict Platform 2 data from Platform 1.

## Compare Model Performance:

Performance is evaluated using Mean Squared Error (MSE) and R-squared (R²).

The true versus predicted values for both models are displayed to visually compare the accuracy of predictions.

## Dependencies:

numpy

pandas

scikit-learn

xgboost

## Example Output:

The script outputs the following:

Performance Metrics for both Lasso regression and XGBoost:

Mean Squared Error (MSE)

R-squared (R²)

Comparison of True vs Predicted Counts for both models.

## Objective:

This code evaluates the accuracy of prediction models for transforming data from one platform (Platform 1) to another (Platform 2) based on genomic counts. 
