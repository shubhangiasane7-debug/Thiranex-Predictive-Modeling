# Predictive Modeling Using Machine Learning

## Project Overview

This project demonstrates predictive modeling using Machine Learning. A Linear Regression model is developed to predict students' Math Scores based on their Study Hours, Attendance, Reading Score, and Writing Score.

## Objective

The main objective of this project is to build and evaluate a supervised machine learning model that can predict student performance.

## Dataset

The project uses a student performance dataset containing 100 student records.

### Features

* Study_Hours
* Attendance
* Reading_Score
* Writing_Score

### Target Variable

* Math_Score

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## Machine Learning Algorithm

Linear Regression is used as the predictive modeling algorithm.

The dataset is divided into:

* 80% training data
* 20% testing data

The model is trained using the training dataset and evaluated using the testing dataset.

## Model Evaluation

The model achieved the following results:

| Metric                         |  Score |
| ------------------------------ | -----: |
| Mean Absolute Error (MAE)      |   0.71 |
| Mean Squared Error (MSE)       |   0.78 |
| Root Mean Squared Error (RMSE) |   0.89 |
| R² Score                       | 0.9912 |

The R² score of approximately 99.12% indicates that the model performs very well on the test dataset.

## Visualization

The project includes:

1. Actual vs Predicted Math Scores
2. Prediction Error Plot

These visualizations help evaluate how closely the predicted values match the actual values.

## Project Structure

```text
Thiranex_Predictive_Modeling/
│
├── predictive_modeling.py
├── student_performance.csv
├── requirements.txt
├── actual_vs_predicted.png
├── prediction_errors.png
└── README.md
```

## How to Run

### 1. Install required libraries

```bash
pip install -r requirements.txt
```

### 2. Run the Python program

```bash
python predictive_modeling.py
```

### 3. View the results

The program displays the model evaluation metrics and generates the prediction visualizations.

## Conclusion

The Linear Regression model successfully predicts Math Scores using student academic and attendance features. The model achieved a strong R² score of 0.9912 on the test dataset, demonstrating good predictive performance for this dataset.
