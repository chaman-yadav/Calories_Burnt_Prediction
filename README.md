# Calories Burnt Prediction using Machine Learning with Python

This project demonstrates a machine learning approach to predict the number of calories burnt during exercise based on various physiological and exercise-related parameters. The model is built using Python and various machine-learning libraries.

Click on the link below to access the deployed app:
[Run the App](https://caloriesburntprediction.streamlit.app/)


## Dataset

The dataset consists of two CSV files: `calories.csv` and `exercise.csv`.

### Exercise Data
| User_ID | Gender | Age | Height | Weight | Duration | Heart_Rate | Body_Temp |
|---------|--------|-----|--------|--------|----------|------------|-----------|
| 14733363 | male | 68 | 190.0 | 94.0 | 29.0 | 105.0 | 40.8 |
| 14861698 | female | 20 | 166.0 | 60.0 | 14.0 | 94.0 | 40.3 |
| ... | ... | ... | ... | ... | ... | ... | ... |

### Calories Data
| User_ID | Calories |
|---------|----------|
| 14733363 | 231.0 |
| 14861698 | 66.0 |
| ... | ... |

The datasets are concatenated based on `User_ID` to form a comprehensive dataset for analysis.


## Libraries Used
- streamlit: To create the web application.
- numpy: For numerical computations.
- pandas: For data manipulation and analysis.
- seaborn: For data visualization.
- matplotlib: For plotting graphs.
- scikit-learn: For building and evaluating machine learning models.
- xgboost: For implementing the XGBoost model.
- pickle: For saving the trained model.

## Exploratory Data Analysis (EDA)

The following visualizations were used in the EDA:

1. Scatter plots to visualize relationships between features:
    - Height vs Weight
    - Duration vs Heart Rate
    - Duration vs Calories

2. Count plot of the `Gender` distribution.

3. Histograms to visualize the distribution of:
    - Age
    - Height
    - Weight

4. Correlation matrix heatmap to understand the relationships between numeric features.

## Data Preparation

- The `Gender` column was mapped to numeric values (`male`: 0, `female`: 1).
- The data was split into training and testing sets using `train_test_split`.

## Model Training

Four regression models were trained and evaluated:
- Linear Regression
- Random Forest Regressor
- Decision Tree Regressor
- XGBRF Regressor

The Random Forest Regressor was selected as the best model based on Mean Squared Error (MSE) and R² score.

## Model Evaluation

| Model | MSE | R² Score |
|-------|-----|----------|
| Linear Regression | 130.09 | 0.967 |
| Random Forest Regressor | 7.20 | 0.998 |
| Decision Tree Regressor | 28.62 | 0.993 |
| XGBRF Regressor | 58.36 | 0.985 |

The Random Forest Regressor was chosen due to its superior performance.

## Model Verification
For model verification, check the ipynb file.

## Application
Click on the link below to access the deployed app on your system:
[Run the App](https://caloriesburntprediction.streamlit.app/)



