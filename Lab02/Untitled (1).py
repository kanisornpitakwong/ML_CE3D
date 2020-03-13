#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install quandl')


# In[3]:


# Stock data
import quandl
import datetime
# Analyzing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics


# In[4]:


df = quandl.get("WIKI/GOOG")


# In[5]:


df


# In[6]:


df = df.dropna()
df = df.iloc[:, 3].values
df


# In[7]:


print(type(df))


# In[8]:


scaler = preprocessing.StandardScaler()
df = scaler.fit_transform(df.reshape(-1,1))
df = pd.DataFrame(df)


# In[9]:


df = df.rename(columns={0: "Close"})


# In[10]:


df


# In[11]:


Next_N_Day = 30
df['GT'] = df[['Close']].shift(-Next_N_Day)


# In[12]:


df = df.dropna()
df


# In[13]:


test_X = df.iloc[-60:, 0].values
test_Y = df.iloc[-60:, 1].values
X = df.iloc[0:len(df)+1,0].values
y = df.iloc[0:len(df)+1,1].values


# In[14]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2,random_state = 42)


# In[15]:


plt.figure(figsize = (20, 7))
plt.scatter(X_train, y_train, marker = 'o', color = 'r')
plt.scatter(X_test, y_test, marker = 'o', color = 'b')
plt.show()







# In[ ]:




