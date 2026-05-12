# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Collection & Preprocessing Import required libraries (pandas, numpy, matplotlib, sklearn). Load the dataset (student_scores.csv). Separate the independent variable (Hours) and dependent variable (Scores). Split the dataset into training and testing sets.
2. Model Training Initialize the Linear Regression model. Train the model using the training dataset (x_train, y_train).
3. Model Prediction & Visualization Predict scores for the test dataset. Plot the regression line with training data (gradient color for points). Plot the regression line with testing data (compare actual vs predicted values).
4. Model Evaluation Calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE). Display evaluation results to assess model accuracy.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: GERIUS G
RegisterNumber:  212224040090
*/
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('student_scores.csv')

df.head()
df.tail()

X = df.iloc[:, :-1].values
print(*X)

Y = df.iloc[:, 1].values
print(*Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/3, random_state=0
)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)
print(*Y_pred)
print(*Y_test)

# Training set
plt.scatter(X_train, Y_train, color="orange")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Testing set
plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, regressor.predict(X_test), color="green")
plt.title("Testing set (Hours vs Scores)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Evaluation
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
```

## Output:
<img width="815" height="242" alt="image" src="https://github.com/user-attachments/assets/b69600d8-4ee1-4e63-9331-212822c57e61" />
<img width="1094" height="704" alt="image" src="https://github.com/user-attachments/assets/12740d74-efb9-442c-9c20-a0d7a35ca227" />
<img width="1213" height="701" alt="image" src="https://github.com/user-attachments/assets/aadea9ba-d4d1-4a86-9c32-0846fc430240" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
