#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary libraries
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.cluster import KMeans 
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


#load csv data 
country_data=pd.read_csv(r"C:\Users\HP\OneDrive\Documents\DATA SETS\country_data.csv",delimiter=";")


# In[3]:


country_data.head()


# In[5]:


#some EDA
country_data.describe()


# In[6]:


country_data.dtypes


# In[8]:


#clean the data by removing columns with nan, -ve and other unwanted characters or values.
#drop the slope first and second wave and store the data as count_data.
count_data=country_data.drop(['Slope first wave', 'Slope second wave'], axis=1)


# In[9]:


count_data.head()


# In[11]:


#country name is an object type data which is not suitale for clustering model, so it's dropped.
C_data = count_data.drop('Country Name', axis=1)


# In[12]:


C_data.head()


# In[17]:


#Normalization of data
#returns the data as a numpy array which is suitable for the model and stored as Clus_dataSet
from sklearn.preprocessing import StandardScaler
X = C_data.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet[0:5]


# In[18]:


#modeling using k-means clustering 
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)


# In[19]:


#assign the label to each row in data frame 
C_data["Clus_km"] = labels
C_data.head(5)


# In[20]:


#check centroid values 
C_data.groupby('Clus_km').mean()


# In[21]:


#distribution of countries based on GDP for health and demographic index
area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('GDP FOR HEALTH', fontsize=18)
plt.ylabel('Demographic Index', fontsize=16)

plt.show()


# In[22]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('GDP FOR HEALTH', fontsize=18)
# plt.xlabel('Demographic Index', fontsize=16)
# plt.zlabel('10% Highest Income 2018', fontsize=16)
ax.set_xlabel('GDP FOR HEALTH')
ax.set_ylabel('Demographic Index')
ax.set_zlabel('10% Highest Income 2018')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))


# In[59]:


# plt.ylabel(First wave max R_o', fontsize=18)
# plt.xlabel('Second wave max R_o', fontsize=16)

area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('First wave max R_o', fontsize=18)
plt.ylabel('Second wave max R_o', fontsize=16)

plt.show()


# In[ ]:




