# Missing Indicator

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import MissingIndicator,SimpleImputer

df = pd.read_csv('tit_train.csv',usecols=['Age','Fare','Survived'])

df.head()

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

X_train.head()

si = SimpleImputer()
X_train_trf = si.fit_transform(X_train)
X_test_trf = si.transform(X_test)

X_train_trf

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train_trf,y_train)

y_pred = clf.predict(X_test_trf)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

mi = MissingIndicator()

mi.fit(X_train)

mi.features_

X_train_missing = mi.transform(X_train)

X_train_missing

X_test_missing = mi.transform(X_test)

X_test_missing

X_train['Age_NA'] = X_train_missing

X_test

X_test['Age_NA'] = X_test_missing

X_train

si = SimpleImputer()

X_train_trf2 = si.fit_transform(X_train)
X_test_trf2 = si.transform(X_test)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train_trf2,y_train)

y_pred = clf.predict(X_test_trf2)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

si = SimpleImputer(add_indicator=True)

X_train = si.fit_transform(X_train)

X_test = si.transform(X_test)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train_trf2,y_train)

y_pred = clf.predict(X_test_trf2)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

