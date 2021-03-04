#!/usr/bin/env python
# coding: utf-8

# BELOW IS A PREDICTIVE MODEL USING PYCARET AND BOSTON HOUSING DATA

# In[1]:


import pycaret
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.datasets import get_data


# In[2]:


df=get_data('boston')


# **Data Preprocessing
# 
# This data is already cleaned from Kaggle.
# 

# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# the above shows that the datasets are complete and no
# missing values.

# In[5]:


df.rename(columns={"medv":"Price"}, inplace=True)


# In[6]:


df.head(5)


# EXPLORATORY DATA ANALYSIS

# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


#TO FIND OUT THE CORRELATION BETWEEN THE FEATURES

corr=df.corr()

corr.shape


# In[10]:


#plotting the heatmap of correlation between features  


# In[11]:




plt.figure(figsize=(14,14))

sns.heatmap(corr,cbar=False,square=True,
            fmt=".2%",annot=True,cmap="Greens")


# In[12]:


#checking null values using heatmap 


sns.heatmap(df.isnull(),yticklabels=False,cmap='viridis')


# In[13]:


#from the plot above it appears there are no null values. 


# In[14]:


#plot count plot for each feature 


# In[15]:


sns.set_style("whitegrid")
print("count for rad value \n", sns.countplot(x="rad",data=df))


# In[16]:


sns.set_style("whitegrid")
print("count for chas feature \n", sns.countplot(x="chas",data=df))


# In[17]:


sns.set_style("whitegrid")
sns.countplot(x="chas",hue='rad',data=df,palette='RdBu_r')
print("Chas Data")


# In[18]:


#understanding house's age feature using distplot


# In[19]:


sns.distplot(df['age'].dropna(),kde=False,color='blue',bins=40)
print("House Age feature understanding")


# In[20]:


sns.distplot(df['crim'].dropna(),kde=False,color='blue',bins=40)
print("crim rate ")


# In[21]:


sns.distplot(df['rm'].dropna(),kde=False,color='blue',bins=40)
print("Number of rooms in the house")


# In[34]:


#doing regression with pycaret
from pycaret.regression import*
reg=setup(data=df,target="Price")


# In[35]:


#first compare models 
compare_models()


# In[36]:


best = compare_models(sort = 'R2')


# In[37]:


#create model, the best model has been displayed from the comparison made


# In[38]:


#et=extra tress regressor
et=create_model('et')


# In[41]:


#next tune model


# In[42]:


tuned_et=tune_model(et)


# In[43]:


#next plot model


# In[44]:


#residual plot for extra tree regressor model 
plot_model(tuned_et)


# In[45]:


#prediction error plot
plot_model(tuned_et,plot='error')


# In[46]:


#feature importance plot 
plot_model(tuned_et,plot='feature')


# In[55]:


#Predict on Test / Hold-out Sample

predict_model(tuned_et);


# In[56]:


#finalize model for deployment

final_et=finalize_model(tuned_et)


# In[57]:


print(final_et)


# In[58]:


predict_model(final_et);


# In[61]:


unseen_data=df*0.1


# In[62]:


unseen_data


# In[65]:


unseen_prediction=predict_model(final_et,data=unseen_data)
unseen_prediction.head()


# In[ ]:




