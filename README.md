# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all necessary packages and dataset: Import the required libraries (like pandas, numpy, and sklearn) and load the dataset needed to implement Logistic Regression.

2.Clean the dataset: Create a copy of the actual dataset and remove any fields that are unnecessary for the analysis.

3.Define variables: Select the dependent variable (target) and independent variables (features) from the processed dataset.

4.Perform Logistic Regression: Execute the Logistic Regression algorithm on the prepared data.

5.Evaluate results: Print the values for the confusion matrix, accuracy score, and classification report to determine whether the student is placed or not.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: R.GOKULAVANI
RegisterNumber:  212225220035
*/
```
import pandas as pd
import numpy as np

# Loading the dataset
df = pd.read_csv("C:\Placement_Data.csv")
df

# Creating a copy and preprocessing
df1 = df.copy()
df1
df1 = df1.drop(['sl_no', 'salary'], axis=1)
df1.isnull().sum()
df1.duplicated().sum()
df1

# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1['gender'] = le.fit_transform(df1['gender'])
df1['ssc_b'] = le.fit_transform(df1['ssc_b'])
df1['hsc_b'] = le.fit_transform(df1['hsc_b'])
df1['hsc_s'] = le.fit_transform(df1['hsc_s'])
df1['degree_t'] = le.fit_transform(df1['degree_t'])
df1['workex'] = le.fit_transform(df1['workex'])
df1['specialisation'] = le.fit_transform(df1['specialisation'])
df1['status'] = le.fit_transform(df1['status'])
df1

# Splitting features and target
x = df1.iloc[:, :-1]
x
y = df1['status']
y

# Train-Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Model Training
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver="liblinear")
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# Evaluation Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("\nConfusion Matrix:\n", confusion)
print("\nClassification Report:\n", cr)

# Plotting the Confusion Matrix
from sklearn import metrics
cn_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=['true', 'false'])
cn_display.plot()
## Output:
<img width="634" height="345" alt="image" src="https://github.com/user-attachments/assets/6df078f9-bb8c-427a-b89a-2f3a66e1f532" />
<img width="880" height="582" alt="image" src="https://github.com/user-attachments/assets/cab257b3-8c1b-4812-93dc-ca59eb4cfed3" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
