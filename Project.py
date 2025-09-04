import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("StudentPerformanceFactors_withNames_Enrollment.csv")

features = [
            'Attendance', 'Previous_Scores'
]
target = 'Exam_Score'

# Encode categorical variables
df_encoded = pd.get_dummies(df[features + [target]], drop_first=True)
df_model = df_encoded.dropna()

X = df_model.drop(target, axis=1)
y = df_model[target]

# Training test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Training model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate F1 error percentage (after making the scores binary, pass/fail at threshold 40)
threshold = 40
y_test_bin = (y_test >= threshold).astype(int)
y_pred_bin = (y_pred >= threshold).astype(int)

'''
from sklearn.metrics import mean_absolute_percentage_error

# Mean Absolute Percentage Error
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
'''

while(1): #check if roll no. is valid or not
    idx = int(input("Enter roll no. of the student: ")) #Between 1-6607
    if idx<1 or idx>6607:
        print('INVALID ROLL NO.!!!\n ENTER AGAIN!!')
        continue
    else:
        break

idx -=1
student_row = df.iloc[idx][features]
student_df = pd.DataFrame([student_row])
student_encoded = pd.get_dummies(student_df)
student_encoded = student_encoded.reindex(columns=X.columns, fill_value=0)
predicted_score = model.predict(student_encoded)

print(f'Name: {df.iloc[idx]['Name']}\tGender: {df.iloc[idx]['Gender']}\tRoll no.: {idx+1}\nAttendence: {df.iloc[idx]['Attendance']}\tEnrollment no: {df.iloc[idx]['Enrollment_No']}\tPrevious Score: {df.iloc[idx]['Previous_Scores']}')
print(f"Predicted Exam Score for student at index {idx}: {predicted_score[0]:.2f}")
