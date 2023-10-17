# Z - score outlier detection and its removal - Feature Engineering - Day 10

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('placement.csv')

df.shape

df.sample(5)

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
sns.distplot(df['cgpa'])
plt.subplot(1,2,2)
sns.distplot(df['placement_exam_marks'])
plt.show()

df['placement_exam_marks'].skew()

print("Mean value of CGPA :",df['cgpa'].mean())
print("Std value of CGPA :",df['cgpa'].std())
print("Min value of CGPA :",df['cgpa'].min())
print("Max value of CGPA :",df['cgpa'].max())

# Finding the boundary or range[-3sigma,3sigma] values

print("Highest boundary value :",df['cgpa'].mean()+3*df['cgpa'].std())

print("Lowest boundary value :",df['cgpa'].mean()-3*df['cgpa'].std())

# Finding outliers
df[(df['cgpa']>8.80)|(df['cgpa']<5.11)]

# Approcah 1
# Trimming

df0=df[(df['cgpa']<8.80)&(df['cgpa']>5.11)]
df0

# Approach 2
# calculation of Z-score

df['cgpa_Zscore'] = (df['cgpa']-df['cgpa'].mean())/df['cgpa'].std()

df.head()

df[df['cgpa_Zscore'] > 3]

df[df['cgpa_Zscore'] < -3]

df[(df['cgpa_Zscore'] > 3) | (df['cgpa_Zscore'] < -3)]

# Trimming 
new_df = df[(df['cgpa_Zscore'] < 3) & (df['cgpa_Zscore'] > -3)]

new_df

# Capping

upper_limit = df['cgpa'].mean() + 3*df['cgpa'].std()
lower_limit = df['cgpa'].mean() - 3*df['cgpa'].std()

lower_limit

df['cgpa'] = np.where(
    df['cgpa']>upper_limit,
    upper_limit,
    np.where(
        df['cgpa']<lower_limit,
        lower_limit,
        df['cgpa']
    )
)

df.shape

df['cgpa'].describe()