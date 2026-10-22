# MMA3001 Project – Concrete Compressive Strength Prediction

## Project Overview

This project investigates the use of machine learning to predict the compressive strength of concrete based on its mixture composition and age.

The aim is to develop, optimise and compare two regression models to evaluate how accurately concrete compressive strength can be predicted from experimental data.

The project uses Linear Regression as a baseline model and Decision Tree Regression as a nonlinear alternative. Model performance is compared using statistical evaluation metrics and predictions on unseen test data.

## Dataset

The project uses the [UCI Concrete Compressive Strength Dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength).

The dataset contains 1,030 observations, with eight input variables:

- Cement (kg/m³)
- Blast furnace slag (kg/m³)
- Fly ash (kg/m³)
- Water (kg/m³)
- Superplasticizer (kg/m³)
- Coarse aggregate (kg/m³)
- Fine aggregate (kg/m³)
- Age (days)

The output variable is concrete compressive strength, measured in MPa.

Initial inspection confirmed that the dataset contains no missing values.

The dataset was divided into training (60%), validation (20%) and testing (20%) subsets using a fixed random seed of 42 for reproducibility.

## Machine Learning Methods

Two regression models were developed and compared:

1. **Linear Regression:** A baseline model used to predict concrete compressive strength by assuming a linear relationship between the input variables and the target variable.
2. **Decision Tree Regression:** A nonlinear regression model capable of identifying more complex relationships between concrete mixture composition, age and compressive strength.

Model performance was evaluated using:

- R² (coefficient of determination)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

A sensitivity analysis investigated the effect of maximum tree depth on the Decision Tree model's performance.

Tree depths from 1 to 20 were evaluated using training and validation RMSE. A maximum depth of 12 was selected because it produced the lowest validation RMSE of 6.1469 MPa.

The optimised Decision Tree and Linear Regression models were subsequently evaluated on the held-out test dataset.

Additional analysis included actual versus predicted strength plots, residual analysis and Decision Tree feature importance.

## Software and Tools

The project was developed using:

- Python
- Google Colab
- NumPy
- pandas
- scikit-learn
- Matplotlib
- pytest
- pdoc

## Running the Project

The main project notebook is `Concrete_Strength_Prediction.ipynb`.

To run the project:

1. Open `Concrete_Strength_Prediction.ipynb` in Google Colab.
2. Ensure the required Python libraries are available.
3. Run the notebook cells sequentially using **Runtime → Run all**.
4. The concrete dataset is loaded directly from the GitHub repository.
5. The notebook performs data preparation, model training, optimisation, evaluation and visualisation.

The main analysis uses NumPy, pandas, scikit-learn and Matplotlib.

The additional tools `pytest` and `pdoc` are used for automated testing and HTML documentation.

## Results

The final model comparison was conducted using the held-out test dataset.

| Performance Metric | Linear Regression | Optimised Decision Tree |
|---|---|---|
| R² | 0.6263 | 0.7766 |
| MAE (MPa) | 7.7392 | 5.1536 |
| RMSE (MPa) | 9.8127 | 7.5867 |

The optimised Decision Tree outperformed Linear Regression across all three performance metrics.

Compared with Linear Regression, the optimised Decision Tree achieved approximately:

- 33.4% lower MAE
- 22.7% lower RMSE

These results indicate that Decision Tree Regression was better able to capture the nonlinear relationships between the concrete input variables and compressive strength for this dataset.

However, the model's prediction errors indicate that it should not be used as a substitute for experimental strength testing in safety-critical engineering applications without further validation.

## Code Functions and Documentation

Three reusable Python functions were developed and stored in `model_utils.py`:

1. `evaluate_model()` — calculates R², MAE and RMSE.
2. `plot_predictions()` — generates actual versus predicted strength plots.
3. `analyse_tree_depth()` — evaluates Decision Tree performance across different maximum depths and identifies the depth with the lowest validation RMSE.

Each function includes a NumPy-style docstring describing its purpose, parameters and return values.

HTML documentation was generated using `pdoc` and saved as `model_utils.html`.

## Automated Testing

Automated unit tests were developed using the `pytest` framework and stored in `test_model_utils.py`.

The tests verify that:

- The evaluation function calculates correct performance metrics for known predictions.
- The plotting function successfully generates a figure.
- The tree-depth analysis returns valid results and identifies the minimum-validation-error depth.

All three automated tests passed.

To run the tests in an environment containing the required dependencies, execute:

```bash
python -m pytest -v test_model_utils.py
```

The saved test output is available in `test_report.txt`.

## AI Acknowledgement Statement

This project uses Artificial Intelligence (AI) tools, including ChatGPT, for assistance with conceptual understanding, Python coding, debugging, documentation and project planning.

AI-assisted material has been reviewed, tested and adapted by the author. Specific uses of AI will be documented where appropriate, in accordance with the unit's requirements.

## Licence

See the `LICENSE` file for the repository's software licence. The dataset remains subject to its original source licence and attribution requirements.
