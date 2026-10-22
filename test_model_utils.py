
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from model_utils import (
    evaluate_model,
    plot_predictions,
    analyse_tree_depth
)


def test_evaluate_model():
    """Check regression metrics for perfect predictions."""
    actual = [10, 20, 30]
    predicted = [10, 20, 30]

    r2, mae, rmse = evaluate_model(actual, predicted)

    assert r2 == pytest.approx(1.0)
    assert mae == pytest.approx(0.0)
    assert rmse == pytest.approx(0.0)


def test_plot_predictions():
    """Check that the plotting function creates a figure."""
    plt.close("all")

    plot_predictions(
        [10, 20, 30],
        [12, 18, 29],
        "Test Model"
    )

    assert len(plt.get_fignums()) == 1
    plt.close("all")


def test_analyse_tree_depth():
    """Check that depth analysis returns valid results."""
    X_train = np.arange(20).reshape(-1, 1)
    y_train = np.arange(20, dtype=float)

    X_val = np.array([[2.5], [7.5], [12.5], [17.5]])
    y_val = np.array([2.5, 7.5, 12.5, 17.5])

    results, best_depth = analyse_tree_depth(
        X_train, y_train, X_val, y_val, max_depth=5
    )

    assert len(results) == 5
    assert best_depth == int(
        results.loc[
            results["Validation RMSE (MPa)"].idxmin(),
            "Max Depth"
        ]
    )
    assert 1 <= best_depth <= 5
