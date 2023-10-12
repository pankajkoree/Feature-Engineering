# Handling Missing Categorical Values using frequent value imputation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('train.csv')

df=df[['GarageQual','FireplaceQu','SalePrice']]

df.head()

df.isnull().mean()*100

df['GarageQual'].value_counts().plot(kind='bar')

df['GarageQual'].mode()

ax=plt.subplot(111)

df[df['GarageQual'] == 'TA']['SalePrice'].plot(kind='kde')
df[df['GarageQual'].isnull()]['SalePrice'].plot(kind='kde',color='red')

lines, labesls = ax.get_legend_handles_labels()
labesls = ['Houses with TA','Houses with NA']
ax.legend(lines, labesls, loc='best')

plt.title('GarageQual')
plt.show()

temp = df[df['GarageQual']=='TA']['SalePrice']

df['GarageQual'].fillna('TA', inplace=True)

df['GarageQual'].value_counts().plot(kind='bar')

ax=plt.subplot(111)

temp.plot(kind='kde')

# distribution of the variable after imputation
df[df['GarageQual']=='TA']['SalePrice'].plot(kind='kde')

lines,labels = ax.get_legend_handles_labels()
labels = ['Original Value','Imputed variable']
ax.legend(lines,labels)

plt.title("GarageQual")
plt.show()

df['FireplaceQu'].value_counts().plot(kind='bar')

df['FireplaceQu'].mode()

fig = plt.figure()
ax = fig.add_subplot(111)

df[df['FireplaceQu']=='Gd']['SalePrice'].plot(kind='kde', ax=ax)

df[df['FireplaceQu'].isnull()]['SalePrice'].plot(kind='kde', ax=ax, color='red')

lines, labels = ax.get_legend_handles_labels()
labels = ['Houses with Gd', 'Houses with NA']
ax.legend(lines, labels, loc='best')

plt.title('FireplaceQu')
plt.show()

temp = df[df['FireplaceQu']=='Gd']['SalePrice']

df['FireplaceQu'].fillna('Gd', inplace=True)

df['FireplaceQu'].value_counts().plot(kind='bar')

fig = plt.figure()
ax = fig.add_subplot(111)


temp.plot(kind='kde', ax=ax)

# distribution of the variable after imputation
df[df['FireplaceQu'] == 'Gd']['SalePrice'].plot(kind='kde', ax=ax, color='red')

lines, labels = ax.get_legend_handles_labels()
labels = ['Original variable', 'Imputed variable']
ax.legend(lines, labels, loc='best')

# add title
plt.title('FireplaceQu')
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.drop(columns=['SalePrice']),df['SalePrice'],test_size=0.2)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_train)

imputer.statistics_