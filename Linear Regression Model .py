#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


Real_estate=pd.read_csv(r"C:\Users\HP\Downloads\Housing_data\Real_estate.csv")


# In[3]:


Real_estate.head()


# In[4]:


R_E=Real_estate.rename(columns={"X1 transaction date":"transaction date","X2 house age":"house age",
                               "X3 distance to the nearest MRT station":"distance to the nearest MRT station",
                               "X4 number of convenience stores":"number of convenience stores",
                               "X5 latitude":"latitude",
                               "X6 longitude":"longitude","Y house price of unit area":"house price of unit area"})


# In[5]:


R_E.head()


# In[6]:


R_E.describe()


# In[7]:


R_E.info()


# In[8]:


#select features to explore.
Ex_data=R_E[["house age","distance to the nearest MRT station","number of convenience stores","house price of unit area"]]


# In[9]:


Ex_data.head()


# In[10]:


visualz=Ex_data[["house age","distance to the nearest MRT station","number of convenience stores","house price of unit area"]]
visualz.hist()
visualz.plot()


# In[11]:


#plot each column against price 
import seaborn as sns
sns.regplot(data=Ex_data,x="house age",y="house price of unit area")


# In[12]:


import seaborn as sns
sns.regplot(data=Ex_data,x="distance to the nearest MRT station",y="house price of unit area")


# In[13]:


import seaborn as sns
sns.regplot(data=Ex_data,x="number of convenience stores",y="house price of unit area")


# In[14]:


msk=np.random.rand(len(R_E))<0.8
train=Ex_data[msk]
test=Ex_data[~msk]


# In[15]:


#Train data distribution
plt.scatter(train["distance to the nearest MRT station"],train["house price of unit area"])
plt.xlabel("distance to the nearest MRT station")
plt.ylabel("house price of unit area")
plt.show()


# In[16]:


#modelling
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['distance to the nearest MRT station']])
train_y = np.asanyarray(train[['house price of unit area']])
regr.fit (train_x, train_y)
# The coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)


# In[17]:


plt.scatter(train["distance to the nearest MRT station"], train["house price of unit area"])
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("distance to the nearest MRT")
plt.ylabel("house price per unit of area")


# In[18]:


from sklearn.metrics import r2_score

test_x = np.asanyarray(test[['distance to the nearest MRT station']])
test_y = np.asanyarray(test[['house price of unit area']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )


# In[19]:


test_y_

