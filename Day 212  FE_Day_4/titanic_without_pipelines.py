# Titanic without Pipeline

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('train.csv')

df.head()

df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)

print(df.head())

# Step 1 -> train/test/split
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Survived']),df['Survived'],test_size=0.2,random_state=42)

print(X_train.head(2))

print(y_train.head())

print(df.isnull().sum())

# Applying imputation

si_age = SimpleImputer()
si_embarked = SimpleImputer(strategy='most_frequent')

X_train_age = si_age.fit_transform(X_train[['Age']])
X_train_embarked = si_embarked.fit_transform(X_train[['Embarked']])

X_test_age = si_age.transform(X_test[['Age']])
X_test_embarked = si_embarked.transform(X_test[['Embarked']])

print(X_train_embarked)

# one hot encoding Sex and Embarked

ohe_sex = OneHotEncoder(sparse=False,handle_unknown='ignore')
ohe_embarked = OneHotEncoder(sparse=False,handle_unknown='ignore')

X_train_sex = ohe_sex.fit_transform(X_train[['Sex']])
X_train_embarked = ohe_embarked.fit_transform(X_train_embarked)

X_test_sex = ohe_sex.transform(X_test[['Sex']])
X_test_embarked = ohe_embarked.transform(X_test_embarked)

print(X_train_embarked)

print(X_train.head(2))

X_train_rem = X_train.drop(columns=['Sex','Age','Embarked'])

X_test_rem = X_test.drop(columns=['Sex','Age','Embarked'])

X_train_transformed = np.concatenate((X_train_rem,X_train_age,X_train_sex,X_train_embarked),axis=1)
X_test_transformed = np.concatenate((X_test_rem,X_test_age,X_test_sex,X_test_embarked),axis=1)

print(X_test_transformed.shape)

clf = DecisionTreeClassifier()
clf.fit(X_train_transformed,y_train)

y_pred = clf.predict(X_test_transformed)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

import pickle

pickle.dump(ohe_sex,open('ohe_sex.pkl','wb'))
pickle.dump(ohe_embarked,open('ohe_embarked.pkl','wb'))
pickle.dump(clf,open('clf.pkl','wb'))