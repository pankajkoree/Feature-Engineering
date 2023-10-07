# Column Transformer on Titaninc Dataset

import numpy as np
import pandas as pd

df=pd.read_csv('Titanic-Dataset.csv')

print(df)


df=df[['Sex','Age','Pclass','Embarked']]

print(df.shape)

df=df.iloc[0:100,0:4]

print(df)

print(df.shape)

print(df.isnull().sum())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['Embarked']),df['Embarked'],test_size=0.2)

print(X_train)

print(y_train)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

# applying imputer on age column
si = SimpleImputer()
age_train = si.fit_transform(X_train[['Age']])

age_test = si.fit_transform(X_test[['Age']])

print(age_train.shape)

print(age_train)

print(df)

# applying ordinal encoding on sex

oe = OrdinalEncoder(categories=[['male','female']])

sex_train = oe.fit_transform(X_train[['Sex']])

sex_test = oe.fit_transform(X_test[['Sex']])

print(sex_train.shape)

print(sex_train)

# Onehotencoding on Pclass

ohe = OneHotEncoder(drop='first',sparse=False)

Pclass_train = ohe.fit_transform(X_train[['Pclass']])

Pclass_test = ohe.fit_transform(X_test[['Pclass']])

print(Pclass_train.shape)


X_train_trans = np.concatenate((age_train,sex_train,Pclass_train),axis=1)

X_test_trans = np.concatenate((age_test,sex_test,Pclass_test),axis=1)

print(X_test_trans.shape)

print(X_test_trans)

# - Column transformer

from sklearn.compose import ColumnTransformer

trans = ColumnTransformer(transformers=[
('tnf1',SimpleImputer(),['Age']),
('tnf2',OrdinalEncoder(categories=[['male','female']]),['Sex']),
('tnf3',OneHotEncoder(sparse=False,drop='first'),['Pclass'])
],remainder='passthrough')

print(trans)


trans.fit_transform(X_train)

print(trans.fit_transform(X_train).shape)

