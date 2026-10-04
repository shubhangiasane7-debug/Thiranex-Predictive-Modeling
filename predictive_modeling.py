import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("student_performance.csv")

print("Dataset loaded successfully!")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns)

# Select input features
X = df[['Study_Hours', 'Attendance', 'Reading_Score', 'Writing_Score']]

# Select target variable
y = df['Math_Score']

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Make predictions
y_pred = model.predict(X_test)

print("\nPredicted values:")
print(y_pred)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R² Score:", r2)

# Plot Actual vs Predicted values
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.title("Actual vs Predicted Math Scores")

# Plot prediction errors
errors = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(y_pred, errors)

plt.axhline(y=0)

plt.xlabel("Predicted Math Score")
plt.ylabel("Prediction Error")
plt.title("Prediction Error Plot")

plt.tight_layout()
plt.savefig("prediction_errors.png", dpi=300)
plt.show()

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

# Example prediction for a new student
new_student = pd.DataFrame({
    'Study_Hours': [5],
    'Attendance': [90],
    'Reading_Score': [85],
    'Writing_Score': [84]
})

predicted_score = model.predict(new_student)

print("\n--- New Student Prediction ---")
print("Predicted Math Score:", predicted_score[0])
