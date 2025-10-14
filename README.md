# Sculpture Delivery Pricing Prediction

## 1\. Project Overview

This project aims to develop a machine learning model to accurately predict the shipping cost for sculptures. By analyzing a historical dataset of deliveries, the model learns the relationships between various factors—such as the sculpture's dimensions, material, artist reputation, and shipping logistics—and the final delivery cost.

The primary goal is to create a robust regression model that can generalize well to new, unseen data. The project follows a complete data science pipeline:

  - Data exploration and cleaning.
  - In-depth feature engineering.
  - Preprocessing of categorical and numerical data.
  - Training and evaluation of multiple regression models (Decision Tree and Random Forest).
  - Hyperparameter tuning using GridSearchCV to optimize model performance and prevent overfitting.
  - Final prediction on a new dataset.

The target variable for this regression task is `custo` (cost).


## 2\. Dataset Description

The project utilizes a dataset containing historical shipping records for sculptures. The key features include:

| Column Name           | Description                                                                 | Data Type     |
|-----------------------|-----------------------------------------------------------------------------|---------------|
| `id_cliente`          | Unique identification number for each client.                               | Categorical   |
| `nome_artista`        | The name of the artist who created the sculpture.                           | Categorical   |
| `reputacao_artista`   | The artist's reputation in the market (higher value means higher reputation). | Numerical     |
| `altura`              | The height of the sculpture in meters.                                      | Numerical     |
| `largura`             | The width of the sculpture in meters.                                       | Numerical     |
| `peso`                | The weight of the sculpture in kilograms.                                   | Numerical     |
| `material`            | The material from which the sculpture is made.                              | Categorical   |
| `preco_escultura`     | The price of the sculpture itself.                                          | Numerical     |
| `preco_base_envio`    | The base price for shipping the sculpture.                                  | Numerical     |
| `internacional`       | Indicates if the shipment is international (Yes/No).                        | Categorical   |
| `envio_expresso`      | Indicates if the shipment used express (fast) delivery (Yes/No).            | Categorical   |
| `instalacao_incluida` | Indicates if installation was included with the purchase (Yes/No).          | Categorical   |
| `transporte`          | The mode of transport for the delivery (e.g., Air, Road, Sea).              | Categorical   |
| `fragil`              | Indicates if the item is fragile (Yes/No).                                  | Categorical   |
| `pedido_extra_cliente`| Indicates if the client provided additional delivery details (Yes/No).      | Categorical   |
| `localizacao_remota`  | Indicates if the delivery location is remote (Yes/No).                      | Categorical   |
| `data_agendada`       | The scheduled date for the delivery.                                        | Datetime      |
| `data_entrega`        | The actual date the delivery was made.                                      | Datetime      |
| `custo`               | **(Target Variable)** The final cost of the delivery.                       | Numerical     |


## 3\. Project Pipeline

The project was structured into the following key stages:

### 3.1. Data Loading and Feature Engineering

  - The training data (`entregas.csv`) was loaded into a pandas DataFrame.
  - **Date Transformation**: The `data_agendada` and `data_entrega` columns were converted from object types to datetime objects.
  - **New Features Created**:
      - `diferenca_dias_entrega`: A numerical feature was engineered to represent the difference (in days) between the actual and scheduled delivery dates.
      - Date Components: The day, month, and year were extracted from both date columns to capture potential seasonal or temporal patterns.
  - The original datetime columns were then dropped.

### 3.2. Data Preprocessing

  - **High-Cardinality Features**: `id_cliente` and `nome_artista` were dropped as they are unique identifiers with no predictive value for a general model.
  - **Categorical Encoding**: One-Hot Encoding was applied to all categorical features (e.g., `material`, `transporte`, `internacional`) using `pandas.get_dummies()`. This converts them into a numerical format suitable for machine learning algorithms, with `drop_first=True` to avoid multicollinearity.

### 3.3. Model Building and Evaluation

#### a) Decision Tree Regressor (Base Model)

A `DecisionTreeRegressor` was initially trained as a baseline.

  - **Result**: The model achieved a perfect R² score of **1.0** on the training data but only **0.66** on the test data. This indicated significant **overfitting**, where the model memorized the training data instead of learning general patterns.

#### b) Decision Tree Optimization with GridSearchCV

To address overfitting, `GridSearchCV` was used to find the optimal hyperparameters for the Decision Tree.

  - **Hyperparameters Tuned**: `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_leaf_nodes`.
  - **Result**: The optimized Decision Tree showed much better generalization. The training R² score was **0.87**, and the test R² score improved to **0.75**. This demonstrates a much healthier balance between bias and variance.

#### c) Random Forest Regressor

An ensemble method, `RandomForestRegressor`, was trained to see if a collection of trees could produce a more robust model.

  - **Result**: The base Random Forest performed very well, achieving a test R² of **0.825** and an Out-of-Bag (OOB) score of **0.828**. The OOB score provides a reliable estimate of the model's performance on unseen data without needing a separate test set.

#### d) Random Forest Optimization with GridSearchCV

The Random Forest was also tuned using `GridSearchCV` to further refine its performance.

  - **Hyperparameters Tuned**: `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_leaf_nodes`.
  - **Result**: The optimized Random Forest maintained a strong test R² score of **0.826**, with a slightly better balance, making it the **final selected model** for this project.

### 3.4. Final Prediction

  - The best performing model—the optimized `RandomForestRegressor`—was used to make predictions on the new, unseen test dataset (`teste_entregas.csv`).
  - The predictions were appended as a new `custo` column to the test DataFrame.
  - The final result was saved to a new CSV file named `Precos_entregas.csv`.


## 4\. Model Performance Summary

The table below compares the performance metrics of the different models on the test set.

| Model                               | R² (Test) | MAE (Test)   | RMSE (Test)    | Notes                                    |
|-------------------------------------|-----------|--------------|----------------|------------------------------------------|
| Decision Tree Regressor (Base)      | 0.660     | 694.16       | 1544.65        | Severe overfitting observed.             |
| Decision Tree Regressor (Optimized) | 0.754     | 586.98       | 1315.20        | Overfitting reduced, improved performance. |
| Random Forest Regressor (Base)      | 0.825     | 459.51       | 1108.01        | Strong performance out-of-the-box.       |
| **Random Forest Regressor (Optimized)** | **0.826** | **455.38** | **1106.16** | **Best performing and final model.** |


## 5\. Technologies and Libraries Used

  - **Programming Language**: Python 3
  - **Core Libraries**:
      - **Pandas**: For data manipulation, loading CSV files, and creating DataFrames.
      - **NumPy**: For numerical operations, especially in performance metric calculations.
      - **Scikit-learn**: The primary library for machine learning, used for:
          - `train_test_split` for data splitting.
          - `DecisionTreeRegressor` and `RandomForestRegressor` as the learning algorithms.
          - `GridSearchCV` for hyperparameter tuning.
          - `KFold` and `cross_validate` for robust model validation.
          - `r2_score`, `mean_absolute_error`, `mean_squared_error` for model evaluation.
      - **Matplotlib & Seaborn**: For data visualization, specifically for plotting feature importances.
  - **Tools**:
      - **Jupyter Notebook**: As the interactive development environment for coding and analysis.


## 6\. How to Run the Project

1.  **Clone the repository:**

    ```sh
    git clone [URL_of_your_repository]
    ```

2.  **Install the necessary dependencies.** It is recommended to use a virtual environment.

    ```sh
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

3.  **Place the data files** (`entregas.csv` and `teste_entregas.csv`) in the root directory of the project.

4.  **Run the Jupyter Notebook:**
    Open and run the `Regression_Project.ipynb` file to execute the entire data processing and model training pipeline.

## 7\. Final Output

The project generates a file named **`Precos_entregas.csv`**, which contains the original features from the test dataset along with a new `custo` column containing the model's predictions for the shipping cost.
