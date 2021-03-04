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


# In[2]:


C_Data=pd.read_csv(r"C:\Users\HP\Downloads\archive (2)\data.csv")


# In[3]:


C_Data.head(20)


# In[4]:


New_D=C_Data.replace({'diagnosis': {"M": 4, "B": 2}})


# In[5]:


New_D


# In[6]:


New_D.dtypes


# In[7]:


#let's look at the diagnosis distribution based on compactness_mean and perimeter_worst.
ax = New_D[New_D['diagnosis'] == 4][0:50].plot(kind='scatter', x='compactness_mean', y='perimeter_worst', color='DarkBlue', label='malignant');
New_D[New_D['diagnosis'] == 2][0:50].plot(kind='scatter', x='compactness_mean', y='perimeter_worst', color='Yellow', label='benign', ax=ax);
plt.show()


# In[8]:


#Drop the unnamed column with Nan values
New_D=New_D.drop(['Unnamed: 32'], axis=1)


# In[9]:


New_D


# In[10]:


feature_data=New_D[["diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave points_mean","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"]]
X=np.asarray(feature_data)
X[0:5]


# In[11]:


New_D["diagnosis"]=New_D["diagnosis"].astype("int")
y=np.asarray(New_D["diagnosis"])
y[0:5]


# In[12]:


#split data into training and test dataset

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[13]:


#Model SVM using sickit learn Library and rbf kernel of the svm algorithm  .
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 



# In[14]:


yhat = clf.predict(X_test)
yhat [0:5]


# In[15]:


#evaluation
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


# In[16]:


# Compute confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')


# In[18]:


from sklearn.metrics import f1_score
f1_score(y_test, yhat, average='weighted') 

