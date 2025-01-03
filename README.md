This script simulates genomic count data with pre-treatment and post-treatment measurements on two platforms. It compares the performance of Lasso regression and XGBoost models in predicting data for a new platform (Platform 2) based on both pre-treatment and post-treatment data from an original platform (Platform 1).

Key steps:

Simulates data with correlation between pre- and post-treatment values.
Prepares and splits the data for training and testing, standardizing the features.
Fits Lasso regression and XGBoost models to predict Platform 2 data from Platform 1.
Compares model performance using Mean Squared Error (MSE) and R-squared, and displays true vs. predicted values.
This code evaluates the accuracy of prediction models for transforming data from one platform to another.
