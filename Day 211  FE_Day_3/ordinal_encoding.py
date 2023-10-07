# Ordinal Encoding - Feature Engineering  Day 3

import numpy as np
import pandas as pd

df = pd.read_csv('customer.csv')

print(df.sample(5))

df = df.iloc[:,2:]

print(df.head())

from sklearn.preprocessing import OrdinalEncoder

X_train = df[['review','education']]

print(X_train)

oe = OrdinalEncoder(categories=[['Poor','Average','Good'],['School','UG','PG']])


oe.fit(X_train)

X_train = oe.transform(X_train)

print(X_train)

print(oe.categories_)

print(X_train)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y_train = df['purchased']

le.fit(y_train)

print(le.classes_)

from sklearn.model_selection import train_test_split

y_test = train_test_split(y_train,test_size=0.2,random_state=0)

y_train = le.transform(y_train)
y_test = le.transform(y_test)

print(y_train)
