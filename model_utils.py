import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor


def evaluate_model(y_actual, y_predicted):
    """
    Calculate regression performance metrics.

    Parameters
    ----------
    y_actual : array-like
        Actual concrete strengths in MPa.
    y_predicted : array-like
        Predicted concrete strengths in MPa.

    Returns
    -------
    tuple of float
        R², MAE (MPa), and RMSE (MPa).
    """
    r2 = r2_score(y_actual, y_predicted)
    mae = mean_absolute_error(y_actual, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    return r2, mae, rmse


def plot_predictions(y_actual, y_predicted, model_name):
    """
    Plot actual versus predicted concrete compressive strength.

    Parameters
    ----------
    y_actual : array-like
        Actual concrete strengths in MPa.
    y_predicted : array-like
        Predicted concrete strengths in MPa.
    model_name : str
        Title displayed on the graph.

    Returns
    -------
    None
        Displays a scatter plot.
    """
    axis_min = min(min(y_actual), min(y_predicted))
    axis_max = max(max(y_actual), max(y_predicted))

    plt.figure(figsize=(7, 5))
    plt.scatter(y_actual, y_predicted, alpha=0.6)
    plt.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        "r--",
        label="Perfect Prediction"
    )
    plt.xlabel("Actual Strength (MPa)")
    plt.ylabel("Predicted Strength (MPa)")
    plt.title(model_name)
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    plt.legend()
    plt.grid(True)
    plt.show()


def analyse_tree_depth(X_train, y_train, X_val, y_val, max_depth=20):
    """
    Identify the Decision Tree depth with lowest validation RMSE.

    Parameters
    ----------
    X_train : array-like
        Training input features.
    y_train : array-like
        Training strengths in MPa.
    X_val : array-like
        Validation input features.
    y_val : array-like
        Validation strengths in MPa.
    max_depth : int, optional
        Maximum depth to investigate, default 20.

    Returns
    -------
    results : pandas.DataFrame
        Training and validation RMSE at each depth.
    best_depth : int
        Depth producing the lowest validation RMSE.
    """
    train_rmse = []
    val_rmse = []

    for depth in range(1, max_depth + 1):
        model = DecisionTreeRegressor(
            max_depth=depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        val_predictions = model.predict(X_val)

        train_rmse.append(
            np.sqrt(mean_squared_error(y_train, train_predictions))
        )
        val_rmse.append(
            np.sqrt(mean_squared_error(y_val, val_predictions))
        )

    results = pd.DataFrame({
        "Max Depth": range(1, max_depth + 1),
        "Training RMSE (MPa)": train_rmse,
        "Validation RMSE (MPa)": val_rmse
    })

    best_depth = int(
        results.loc[
            results["Validation RMSE (MPa)"].idxmin(),
            "Max Depth"
        ]
    )

    return results, best_depth
