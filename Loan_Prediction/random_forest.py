import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

train_original = pd.read_csv("train_ctrUa4K.csv")
test_original = pd.read_csv("test_lAUu6dG.csv")
print("Data imported")
train = train_original.copy()
test = test_original.copy()

train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True) #Just for credit history
def missing_vals(data):
    for col in data.columns:
        if data[col].dtypes == object:
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            data[col].fillna(data[col].median(), inplace=True)
            data[col] = (data[col] - data[col].mean())/ data[col].std() #Scaling

missing_vals(train)

train = train.drop('Loan_ID',axis=1) 
test = test.drop('Loan_ID',axis=1)

X = train.drop('Loan_Status', axis=1) 
y = train.Loan_Status

X = pd.get_dummies(X) 
train = pd.get_dummies(train) 
test = pd.get_dummies(test)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model2 = RandomForestClassifier(max_depth=9, n_estimators=350, random_state=42) #gc.best_estimator_
model2.fit(X_train, y_train)

p2 = model2.predict(X_test)
pr2 = precision_score(y_test, p2, average="weighted")
rc2 = recall_score(y_test, p2, average="weighted")
f1_2 = f1_score(y_test, p2, average="weighted")

#print("precision: {}\t recall: {}".format(pr2, rc2))
#print("f1: {}".format(f1_2))

test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)
missing_vals(test)

pred_test = model2.predict(test)
submission = pd.read_csv("sample_submission_49d68Cx.csv")
submission['Loan_Status'] = pred_test
submission['Loan_ID'] = test_original['Loan_ID']

submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)

result = pd.DataFrame(submission, columns=['Loan_ID','Loan_Status'])
result.to_csv('Random_Forest.csv', index=False)