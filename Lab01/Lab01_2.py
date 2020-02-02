#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# In[2]:


df = pd.read_csv('watch_test2_sample.csv')
df['uts'] = pd.to_datetime(df['uts'])


# In[3]:


df = df.drop_duplicates()


# In[4]:


df = df.set_index('uts')
df = df.fillna(df.mean())


# In[5]:


accelerator_df = df[['accelerateX', 'accelerateY', 'accelerateZ']].copy()
accelerator_df


# In[6]:


accelerator_df -= accelerator_df.mean()


# In[7]:


accelerator_df.mean()


# In[8]:


accelerator = accelerator_df.to_numpy()


# In[9]:


cov = accelerator.T.dot(accelerator)/(len(accelerator) - 1)
cov


# In[10]:


eigen_values, eigen_vectors = np.linalg.eig(cov)
eigen_values, eigen_vectors 


# In[11]:


sorted_indexes = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[sorted_indexes]
eigen_vectors = eigen_vectors[:, sorted_indexes]
eigen_values, eigen_vectors


# In[12]:


plt.bar(np.arange(len(eigen_values)), eigen_values)


# In[13]:


np.sqrt(eigen_values)


# In[14]:


eigen_vectors = (eigen_vectors.T * np.sqrt(eigen_values)).T
eigen_vectors


# In[15]:


scale = 2
ev1, ev2, ev3 = eigen_vectors.T * scale


# In[17]:


fig = plt.figure(figsize=(50, 10))

ax = fig.add_subplot(141, projection='3d')

ax.plot(accelerator[:, 0],accelerator[:, 1],accelerator[:, 2],'o',markersize=10,color='green',alpha=0.2)

ax.plot([df['accelerateX'].mean()],[df['accelerateY'].mean()],[df['accelerateZ'].mean()],'o',markersize=10,color='red'
        ,alpha=0.5)

ax.plot([0, ev1[0]], [0, ev1[1]], [0, ev1[2]],color='red', alpha=0.8, lw=2)
ax.plot([0, ev2[0]], [0, ev2[1]], [0, ev2[2]],color='violet', alpha=0.8, lw=2)
ax.plot([0, ev3[0]], [0, ev3[1]], [0, ev3[2]],color='cyan', alpha=0.8, lw=2)

ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')

plt.title('Eigenvector')

ax.view_init(10, 60)

plt.show()


# In[18]:


K = 2
pca = accelerator.dot( eigen_vectors[:, :K])
pca


# In[20]:


sns.heatmap(pca)


# In[ ]:





# In[ ]:




