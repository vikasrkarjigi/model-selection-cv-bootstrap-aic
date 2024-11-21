import numpy as np
import matplotlib.pyplot as plt


class ModelSelector:
    def __init__(self):
        pass

    # Linear Regression
    def fit_linear_regression(self, X, y):
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ (X_with_intercept.T @ y)
        return beta

    def predict_linear_regression(self, X, beta):
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        return X_with_intercept @ beta

    # Ridge Regression
    def fit_ridge_regression(self, X, y, alpha=1.0):
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        I = np.eye(X_with_intercept.shape[1])  # Identity matrix for regularization
        I[0, 0] = 0  # Exclude intercept term from regularization
        beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept + alpha * I) @ (X_with_intercept.T @ y)
        return beta

    def predict_ridge_regression(self, X, beta):
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        return X_with_intercept @ beta

    # Lasso Regression (Coordinate Descent)
    def fit_lasso_regression(self, X, y, alpha=1.0, max_iter=1000, tol=1e-4):
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        beta = np.zeros(X_with_intercept.shape[1])  # Initialize coefficients
        n = X_with_intercept.shape[0]

        for _ in range(max_iter):
            beta_old = beta.copy()
            for j in range(len(beta)):
                X_j = X_with_intercept[:, j]
                residual = y - (X_with_intercept @ beta - X_j * beta[j])
                rho = X_j.T @ residual
                if j == 0:  # No regularization for intercept
                    beta[j] = rho / n
                else:
                    beta[j] = np.sign(rho) * max(abs(rho) - alpha / 2, 0) / (X_j.T @ X_j)

            if np.linalg.norm(beta - beta_old, ord=1) < tol:
                break

        return beta

    def predict_lasso_regression(self, X, beta):
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        return X_with_intercept @ beta

    # Metrics and Validation
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def aic(self, X, y, beta, model_name):
        n = len(y)
        y_pred = self.predict_linear_regression(X, beta)  # All predictors share same function
        mse = self.mean_squared_error(y, y_pred)
        num_params = len(beta)
        aic = n * np.log(mse) + 2 * num_params
        # print(f"AIC ({model_name}): {aic:.2f}")
        return aic

    def k_fold_cross_validation(self, X, y, k=5, model_type="linear", alpha=1.0):
        n = len(y)
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_size = n // k
        mse_list = []

        for i in range(k):
            test_idx = indices[i * fold_size:(i + 1) * fold_size]
            train_idx = np.setdiff1d(indices, test_idx)

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if model_type == "linear":
                beta = self.fit_linear_regression(X_train, y_train)
                y_pred = self.predict_linear_regression(X_test, beta)
            elif model_type == "ridge":
                beta = self.fit_ridge_regression(X_train, y_train, alpha)
                y_pred = self.predict_ridge_regression(X_test, beta)
            elif model_type == "lasso":
                beta = self.fit_lasso_regression(X_train, y_train, alpha)
                y_pred = self.predict_lasso_regression(X_test, beta)
            else:
                raise ValueError("Invalid model_type. Choose from 'linear', 'ridge', or 'lasso'.")

            mse_list.append(self.mean_squared_error(y_test, y_pred))

        return np.mean(mse_list)

    def bootstrapping(self, X, y, num_samples=100, model_type="linear", alpha=1.0):
        n = len(y)
        mse_list = []

        for _ in range(num_samples):
            sample_indices = np.random.choice(np.arange(n), size=n, replace=True)
            oob_indices = np.setdiff1d(np.arange(n), sample_indices)

            if len(oob_indices) == 0:
                continue

            X_sample, y_sample = X[sample_indices], y[sample_indices]
            X_oob, y_oob = X[oob_indices], y[oob_indices]

            if model_type == "linear":
                beta = self.fit_linear_regression(X_sample, y_sample)
                y_pred = self.predict_linear_regression(X_oob, beta)
            elif model_type == "ridge":
                beta = self.fit_ridge_regression(X_sample, y_sample, alpha)
                y_pred = self.predict_ridge_regression(X_oob, beta)
            elif model_type == "lasso":
                beta = self.fit_lasso_regression(X_sample, y_sample, alpha)
                y_pred = self.predict_lasso_regression(X_oob, beta)
            else:
                raise ValueError("Invalid model_type. Choose from 'linear', 'ridge', or 'lasso'.")

            mse_list.append(self.mean_squared_error(y_oob, y_pred))

        return np.mean(mse_list)


    def best_model(self, results):
        """
        Identify and print the best model based on the lowest AIC score.

        Parameters:
            results (dict): A dictionary containing model metrics for comparison.
                            Expected format:
                            {
                                "model_name": {
                                    "k_fold_mse": float,
                                    "bootstrap_mse": float,
                                    "aic_score": float
                                },
                                ...
                            }
        """
        # Determine the best model based on the lowest AIC score
        best_model_name = min(results, key=lambda model: results[model]["aic_score"])
        best_metrics = results[best_model_name]

        # Print the best model and its metrics
        print("\nBest Model Summary:")
        print(f"Best Model: {best_model_name.capitalize()}")
        print(f"K-Fold Cross-Validation MSE: {best_metrics['k_fold_mse']:.4f}")
        print(f"Bootstrapping MSE: {best_metrics['bootstrap_mse']:.4f}")
        print(f"AIC Score: {best_metrics['aic_score']:.4f}")

    def plot_comparison(self, results):
        """
        Visualize the comparison of models using MSE and AIC scores.

        Parameters:
            results (dict): A dictionary containing model metrics for comparison.
        """
        models = list(results.keys())
        k_fold_mse = [results[model]["k_fold_mse"] for model in models]
        bootstrap_mse = [results[model]["bootstrap_mse"] for model in models]
        aic_scores = [results[model]["aic_score"] for model in models]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot for K-Fold MSE
        axes[0].bar(models, k_fold_mse, color='skyblue')
        axes[0].set_title('K-Fold Cross-Validation MSE')
        axes[0].set_ylabel('MSE')
        axes[0].set_xlabel('Model')

        # Plot for Bootstrapping MSE
        axes[1].bar(models, bootstrap_mse, color='lightgreen')
        axes[1].set_title('Bootstrapping MSE')
        axes[1].set_ylabel('MSE')
        axes[1].set_xlabel('Model')

        # Plot for AIC Scores
        axes[2].bar(models, aic_scores, color='lightcoral')
        axes[2].set_title('AIC Scores')
        axes[2].set_ylabel('AIC')
        axes[2].set_xlabel('Model')

        plt.tight_layout()
        plt.show()

    def visualize_model_comparison(self, results):
        models = list(results.keys())
        k_fold_mse = [results[model]["k_fold_mse"] for model in models]
        bootstrap_mse = [results[model]["bootstrap_mse"] for model in models]
        aic_scores = [results[model]["aic_score"] for model in models]

        # Plot MSE comparison
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].bar(models, k_fold_mse, color="skyblue")
        ax[0].set_title("K-Fold Cross-Validation MSE")
        ax[0].set_xlabel("Model")
        ax[0].set_ylabel("MSE")
        for i, v in enumerate(k_fold_mse):
            ax[0].text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')

        ax[1].bar(models, bootstrap_mse, color="lightcoral")
        ax[1].set_title("Bootstrapping MSE")
        ax[1].set_xlabel("Model")
        ax[1].set_ylabel("MSE")
        for i, v in enumerate(bootstrap_mse):
            ax[1].text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        # Plot AIC scores
        plt.figure(figsize=(8, 6))
        plt.bar(models, aic_scores, color="lightgreen")
        plt.title("AIC Score Comparison")
        plt.xlabel("Model")
        plt.ylabel("AIC Score")
        for i, v in enumerate(aic_scores):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')

        plt.show()

# ===================================================================================================================================
