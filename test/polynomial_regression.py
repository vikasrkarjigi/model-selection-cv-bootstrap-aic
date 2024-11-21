import numpy as np
from Project2.model_selection.generic_crossvalidation import ModelSelector

# Generate dataset
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 + 1.5 * X.squeeze() - 0.5 * (X.squeeze() ** 2) + np.random.randn(100) * 5

# Add polynomial features manually
degree = 2
X_poly = np.hstack([X ** i for i in range(1, degree + 1)])

# Model selection
selector = ModelSelector()

k_fold_mse = selector.k_fold_cross_validation(X_poly, y, k=5)
bootstrap_mse = selector.bootstrapping(X_poly, y, num_samples=100)
aic_score = selector.aic(X_poly, y)

print("\nPolynomial Regression Test")
print(f"K-Fold Cross-Validation MSE: {k_fold_mse}")
print(f"Bootstrapping MSE: {bootstrap_mse}")
print(f"AIC Score: {aic_score}")


# import pandas as pd
# import numpy as np
# from Project2.model_selection.generic_crossvalidation import ModelSelector
#
#
# def test_wine_quality():
#     # Load the dataset
#     url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
#     data = pd.read_csv(url, sep=';')
#
#     # Features and target
#     X = data.iloc[:, :-1].values  # Features
#     y = data.iloc[:, -1].values  # Target
#
#     # Initialize ModelSelector
#     selector = ModelSelector()
#
#     # Test all models (Linear, Ridge, Lasso)
#     models = ['linear', 'ridge', 'lasso']
#     alpha = 1.0  # Default alpha for Ridge and Lasso
#
#     results = {}
#
#     for model in models:
#         print(f"Testing {model.capitalize()} Regression...")
#
#         # Perform k-fold cross-validation
#         k_fold_mse = selector.k_fold_cross_validation(X, y, k=5, model_type=model, alpha=alpha)
#         print(f"K-Fold Cross-Validation MSE ({model}): {k_fold_mse:.2f}")
#
#         # Perform bootstrapping
#         bootstrap_mse = selector.bootstrapping(X, y, num_samples=100, model_type=model, alpha=alpha)
#         print(f"Bootstrapping MSE ({model}): {bootstrap_mse:.2f}")
#
#         # Compute AIC
#         beta = None
#         if model == 'linear':
#             beta = selector.fit_linear_regression(X, y)
#         elif model == 'ridge':
#             beta = selector.fit_ridge_regression(X, y, alpha=alpha)
#         elif model == 'lasso':
#             beta = selector.fit_lasso_regression(X, y, alpha=alpha)
#
#         aic_score = selector.aic(X, y, beta, model)
#         print(f"AIC Score ({model}): {aic_score:.2f}")
#
#         # Store the results for comparison
#         results[model] = {
#             'k_fold_mse': k_fold_mse,
#             'bootstrap_mse': bootstrap_mse,
#             'aic_score': aic_score
#         }
#
#     # Determine the best model based on AIC
#     best_model = min(results, key=lambda model: results[model]['aic_score'])
#     print("\nBest Model Based on AIC:", best_model.capitalize())
#     print(f"K-Fold MSE: {results[best_model]['k_fold_mse']:.2f}")
#     print(f"Bootstrapping MSE: {results[best_model]['bootstrap_mse']:.2f}")
#     print(f"AIC Score: {results[best_model]['aic_score']:.2f}")
#
#
# if __name__ == "__main__":
#     test_wine_quality()
