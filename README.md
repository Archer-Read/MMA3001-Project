# MMA3001 Project – Concrete Compressive Strength Prediction

## Project Overview

This project investigates the use of machine learning to predict the compressive strength of concrete based on its mixture composition and age.

The aim is to develop and compare two regression models to determine how accurately concrete compressive strength can be predicted from experimental data.

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

## Machine Learning Methods

Two regression models will be developed and compared:

1. **Linear Regression:** A baseline model used to predict concrete compressive strength.
2. **Decision Tree Regression:** An alternative model capable of capturing nonlinear relationships between the input variables and concrete strength.

Model performance will be evaluated using:

- R² (coefficient of determination)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

A sensitivity analysis will investigate the effect of maximum tree depth on the Decision Tree model's performance.

## Software and Tools

- Python
- Google Colab
- pandas
- scikit-learn
- Matplotlib

## Running the Project

The project notebook is `Concrete_Strength_Prediction.ipynb`.

The notebook is developed and executed using Google Colab. Dataset loading instructions and any required dependencies will be documented as the project progresses.

## Results

Results and model comparisons will be added following model development and evaluation.

## AI Acknowledgement Statement

This project uses Artificial Intelligence (AI) tools, including ChatGPT, for assistance with conceptual understanding, Python coding, debugging, documentation and project planning.

AI-assisted material will be reviewed, tested and adapted by the author. Specific uses of AI will be documented where appropriate, in accordance with the unit's requirements.

## Licence

See the `LICENSE` file for the repository's software licence. The dataset remains subject to its original source licence and attribution requirements.
