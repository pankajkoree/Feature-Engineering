# Handling Data and Time - Feature Engineering - Day 6

import numpy as np
import pandas as pd

date = pd.read_csv('orders.csv')
time = pd.read_csv('messages.csv')

date.head()

time.head()

date.info()

time.info()

# Working with Dates

# Converting to datetime datatype
date['date'] = pd.to_datetime(date['date'])

date.info()

# 1. Extract year

date['date_year'] = date['date'].dt.year

date.sample(5)

# 2. Extract Month

date['date_month_no'] = date['date'].dt.month

date.head()

date['date_month_name'] = date['date'].dt.month_name()

date.head()

# Extract Days

date['date_day'] = date['date'].dt.day

date.head()

# day of week
date['date_dow'] = date['date'].dt.dayofweek

date.head()

# day of week - name

date['date_dow_name'] = date['date'].dt.day_name()

date.drop(columns=['product_id','city_id','orders']).head()

# is weekend?

date['date_is_weekend'] = np.where(date['date_dow_name'].isin(['Sunday', 'Saturday']), 1,0)

date.drop(columns=['product_id','city_id','orders']).head()

# Extract week of the year

date['date_week'] = date['date'].dt.weekday

date.drop(columns=['product_id','city_id','orders']).head()

# Extract Quarter

date['quarter'] = date['date'].dt.quarter

date.drop(columns=['product_id','city_id','orders']).head()

# Extract Semester

date['semester'] = np.where(date['quarter'].isin([1,2]), 1, 2)

date.drop(columns=['product_id','city_id','orders']).head()

# Extract Time elapsed between dates

import datetime

today = datetime.datetime.today()

today

today - date['date']

(today - date['date']).dt.days

time.info()

# Converting to datetime datatype
time['date'] = pd.to_datetime(time['date'])

time.info()

time['hour'] = time['date'].dt.hour
time['min'] = time['date'].dt.minute
time['sec'] = time['date'].dt.second

time.head()

# Extract Time part

time['time'] = time['date'].dt.time

time.head()

# Time difference

today - time['date']

# in seconds

(today - time['date'])/np.timedelta64(1,'s')

# in minutes

(today - time['date'])/np.timedelta64(1,'m')

# in hours

(today - time['date'])/np.timedelta64(1,'h')