#!/usr/bin/env python
# coding: utf-8

# # In this notebook, we will try to load and look at some csv (comma separated values) data

# 
# Import all the needed python pacakges and libaries

# In[1]:


get_ipython().magic(u'tensorflow_version 2.x')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Read the data using pandas
# 

# In[ ]:


#----------DATA READING 
filename = 'https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv'
# read file
csv_data = pd.read_csv(filename, sep=',')


# Data is read as a matrix, let's take a look at the shape of that matrix

# In[3]:


print(csv_data.shape)


# Take a look at the first five data points

# In[4]:


print(csv_data.head())


# Print the previous results transposed so we can see all the columns (features)

# In[5]:


print(csv_data.head().transpose())


# Look at the basic data stats per column

# In[6]:


print(csv_data.describe())


# Take a look at just two features using pandas indexing by columns

# In[7]:


print(csv_data[['latitude', 'longitude']])


# Do a scatter plot of 2 of the columns (latitute on the x-axis, longitude on the y-axis) and use 'total_bedrooms' as the size of the dots.

# In[8]:


sns.relplot(x='latitude', y='longitude', size='total_bedrooms', alpha=0.5, palette='muted', data=csv_data)


# Take the first 1000 data points (using the head(xxxx) function) and plot 5 of features against each other.

# In[9]:


sns.pairplot(csv_data.head(1000)[['longitude', 'latitude', 'total_bedrooms', 'median_house_value', 'population']])


# why is the map messed up?

# In[10]:


sns.pairplot(csv_data.s1000)[['longitude', 'latitude', 'total_bedrooms', 'median_house_value', 'population']])

