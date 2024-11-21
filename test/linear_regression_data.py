import numpy as np
import pandas as pd
from Project2.model_selection.generic_crossvalidation import ModelSelector

# # Generate dataset
# np.random.seed(42)
# X = np.random.rand(100, 1) * 10
# y = 5 + 2 * X.squeeze() + np.random.randn(100) * 2
#
# # Model selection
# selector = ModelSelector()
#
# k_fold_mse = selector.k_fold_cross_validation(X, y, k=5)
# bootstrap_mse = selector.bootstrapping(X, y, num_samples=100)
# aic_score = selector.aic(X, y)
#
# print("Linear Regression Test")
# print(f"K-Fold Cross-Validation MSE: {k_fold_mse}")
# print(f"Bootstrapping MSE: {bootstrap_mse}")
# print(f"AIC Score: {aic_score}")
#
#
#
# def test_boston_housing():
#     # Load Boston Housing Dataset
#     url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
#     data = pd.read_csv(url)
#
#     # Features and target
#     X = data.iloc[:, :-1].values  # All columns except the last one
#     y = data.iloc[:, -1].values   # Median value of owner-occupied homes
#
#     # Initialize ModelSelector
#     selector = ModelSelector()
#
#     # Perform k-fold cross-validation
#     k_fold_mse = selector.k_fold_cross_validation(X, y, k=5)
#
#     # Perform bootstrapping
#     bootstrap_mse = selector.bootstrapping(X, y, num_samples=100)
#
#     # Compute AIC
#     aic_score = selector.aic(X, y)
#
#     # Print results
#     print("Boston Housing Dataset Test")
#     print(f"K-Fold Cross-Validation MSE: {k_fold_mse}")
#     print(f"Bootstrapping MSE: {bootstrap_mse}")
#     print(f"AIC Score: {aic_score}")
#
#
# if __name__ == "__main__":
#     test_boston_housing()


def test_wine_quality():
    # Load the dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, sep=';')

    # Features and target
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Target

    # Initialize ModelSelector
    selector = ModelSelector()

    # Model configurations
    models = ["linear", "ridge", "lasso"]
    alpha = 1.0  # Regularization parameter for ridge and lasso
    results = {}  # To store results of each model
    print("Wine Quality Dataset Test")
    for model in models:
        # Perform k-fold cross-validation
        k_fold_mse = selector.k_fold_cross_validation(X, y, k=5, model_type=model, alpha=alpha)

        # Perform bootstrapping
        bootstrap_mse = selector.bootstrapping(X, y, num_samples=100, model_type=model, alpha=alpha)

        # Fit the model to calculate AIC
        if model == "linear":
            beta = selector.fit_linear_regression(X, y)
            y_pred_function = selector.predict_linear_regression
        elif model == "ridge":
            beta = selector.fit_ridge_regression(X, y, alpha)
            y_pred_function = selector.predict_ridge_regression
        elif model == "lasso":
            beta = selector.fit_lasso_regression(X, y, alpha)
            y_pred_function = selector.predict_lasso_regression

        # Compute AIC
        aic_score = selector.aic(X, y, beta, model)

        # Print results
        print(f"\nModel: {model.capitalize()}")
        print(f"K-Fold Cross-Validation MSE: {k_fold_mse:.4f}")
        print(f"Bootstrapping MSE: {bootstrap_mse:.4f}")
        print(f"AIC Score: {aic_score:.4f}")

        results[model] = {
            "k_fold_mse": k_fold_mse,
            "bootstrap_mse": bootstrap_mse,
            "aic_score": aic_score
        }

    # Identify and print the best model
    selector.best_model(results)
    selector.plot_comparison(results)
    selector.visualize_model_comparison(results)

if __name__ == "__main__":
    test_wine_quality()
