#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[2]:


churn_d=pd.read_csv(r"C:\Users\HP\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[3]:


churn_d.head()


# In[4]:


churn_d.dtypes


# In[5]:


churn_d.describe()


# In[6]:


churn_d.head()


# In[11]:


churn_d["gender"]=pd.get_dummies(churn_d["gender"])


# In[13]:


churn_d.head(5)


# In[15]:


churn_d["PhoneService"]=pd.get_dummies(churn_d["PhoneService"])


# In[16]:


churn_d.head()


# In[17]:


churn_d["MultipleLines"]=pd.get_dummies(churn_d["MultipleLines"])


# In[18]:


churn_d.head()


# In[23]:


churn_d["InternetService"]=pd.get_dummies(churn_d["InternetService"])


# In[24]:


churn_d.head()


# In[36]:


churn_d["Churn"]=pd.get_dummies(churn_d["Churn"])


# In[37]:


#select modeling data
churn_d = churn_d[['gender','SeniorCitizen','tenure','PhoneService','InternetService','MultipleLines','MonthlyCharges','TotalCharges','Churn']]
churn_d['Churn'] = churn_d['Churn'].astype('int')
churn_d.head()


# In[41]:


# define X AND y for the data
X=np.asarray(churn_d[['SeniorCitizen','tenure','PhoneService','InternetService','MultipleLines']])
X[0:5]                      
                      


# In[44]:


y=np.asarray(churn_d["Churn"])
y[0:5]


# In[45]:


#normalize the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[46]:


#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[47]:


#build the logisitc regression with sklearn for classification
LogReg=LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LogReg


# In[49]:


yhat=LogReg.predict(X_test)
yhat[0:10]


# In[50]:


#predict proba
yhat_prob = LogReg.predict_proba(X_test)
yhat_prob


# In[51]:


#model evaluation 
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))


# In[52]:


cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')


# In[53]:


print (classification_report(y_test, yhat))


# In[54]:


#using log loss for evaluation
from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)


# In[ ]:




