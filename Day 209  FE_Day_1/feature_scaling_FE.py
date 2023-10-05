# Feature Engineering (FE) - Feature Scaling 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


url = "https://raw.githubusercontent.com/codejust4U/dataset/main/Social_Network_Ads.csv"
df = pd.read_csv(url)


print(df)

df = df.iloc[:,2:]

print(df)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(df.drop('Purchased',axis=1),df['Purchased'],test_size=0.3,random_state=0)

print(X_train.shape,X_test.shape)

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

scalar.fit(X_train)

X_train_Scaled = scalar.transform(X_train)
X_test_scaled = scalar.transform(X_test)

print(scalar.mean_)

X_train_Scaled = pd.DataFrame(X_train_Scaled,columns=X_train.columns)
X_test_Scaled = pd.DataFrame(X_test_scaled,columns=X_test.columns)

print(np.round(X_train.describe(),1))

print(np.round(X_train_Scaled.describe(),1))

# - Effect of Scaling

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

ax1.scatter(X_train['Age'], X_train['EstimatedSalary'])
ax1.set_title("Before Scaling")
ax2.scatter(X_train_Scaled['Age'], X_train_Scaled['EstimatedSalary'],color='red')
ax2.set_title("After Scaling")
plt.show()

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Before Scaling')
sns.kdeplot(X_train['Age'], ax=ax1)
sns.kdeplot(X_train['EstimatedSalary'], ax=ax1)

# after scaling
ax2.set_title('After Standard Scaling')
sns.kdeplot(X_train_Scaled['Age'], ax=ax2)
sns.kdeplot(X_train_Scaled['EstimatedSalary'], ax=ax2)
plt.show()

# - Comparison of Distributions

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Age Distribution Before Scaling')
sns.kdeplot(X_train['Age'], ax=ax1)

# after scaling
ax2.set_title('Age Distribution After Standard Scaling')
sns.kdeplot(X_train_Scaled['Age'], ax=ax2)
plt.show()



fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

# before scaling
ax1.set_title('Salary Distribution Before Scaling')
sns.kdeplot(X_train['EstimatedSalary'], ax=ax1)

# after scaling
ax2.set_title('Salary Distribution Standard Scaling')
sns.kdeplot(X_train_Scaled['EstimatedSalary'], ax=ax2)
plt.show()


# - Why scaling is important?

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr_scaled = LogisticRegression()

lr.fit(X_train,y_train)
lr_scaled.fit(X_train_Scaled,y_train)

y_pred = lr.predict(X_test)
y_pred_scaled = lr_scaled.predict(X_test_scaled)

from sklearn.metrics import accuracy_score

print("Actual",accuracy_score(y_test,y_pred))
print("Scaled",accuracy_score(y_test,y_pred_scaled))

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt_scaled = DecisionTreeClassifier()

dt.fit(X_train,y_train)
dt_scaled.fit(X_train_Scaled,y_train)

y_pred = dt.predict(X_test)
y_pred_scaled = dt_scaled.predict(X_test_scaled)

print("Actual",accuracy_score(y_test,y_pred))
print("Scaled",accuracy_score(y_test,y_pred_scaled))

print(df.describe())

# - Effect of Outlier

df1=pd.DataFrame({'Age':[5],'EstimatedSalary':[1000],'Purchased':[0]})

print(df1)

df1=pd.DataFrame({'Age':[90],'EstimatedSalary':[250000],'Purchased':[1]})

print(df1)

df1=pd.DataFrame({'Age':[95],'EstimatedSalary':[350000],'Purchased':[1]})

print(df1)

df.loc[len(df)] = {'Age': 5, 'EstimatedSalary': 1000,'Purchased':0}
df.loc[len(df)] = {'Age': 90, 'EstimatedSalary': 50000,'Purchased':1}
df.loc[len(df)] = {'Age': 95, 'EstimatedSalary': 350000,'Purchased':1}

print(df)

plt.scatter(df['Age'], df['EstimatedSalary'])
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Purchased', axis=1),
df['Purchased'],test_size=0.3,random_state=0)

X_train.shape, X_test.shape

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# fit the scaler to the train set, it will learn the parameters
scaler.fit(X_train)

# transform train and test sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

ax1.scatter(X_train['Age'], X_train['EstimatedSalary'])
ax1.set_title("Before Scaling")
ax2.scatter(X_train_scaled['Age'], X_train_scaled['EstimatedSalary'],color='red')
ax2.set_title("After Scaling")
plt.show()