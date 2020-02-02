#!/usr/bin/env python
# coding: utf-8

# ### 1.1 ขั้นตอนการทดลองในการนำเข้าข้อมูล แก้ปัญหาข้อผิดพลาดของข้อมูล และการปรับช่วงค่าของข้อมูล

# #### import lib

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# #### read data file

# In[2]:


df = pd.read_csv('watch_test2_sample.csv')


# #### แปลงชนิดของข้อมูล และทำการcopyข้อมูลไว้เพื่อใช้งานในการplot gps จากค่าจริง

# In[3]:


df['uts'] = pd.to_datetime(df['uts'])
df.sort_values('uts', inplace = True)
df_copy = df.copy()


# In[4]:


df.info()


# #### Data Cleaning
#     1.1)จัดการข้อมูลซ้ำซ้อน

# In[5]:


df = df.drop_duplicates(keep="first")
df.info()


# #### Data Cleaning 
#     1.2)จัดการข้อมูลหายโดยการแทนที่ค่าmedianของแต่ละ column ลงไป

# In[6]:


df = df.fillna(df.median())
df


# #### Data Cleaning
#     2.1) ทำการแทรกข้อมูลที่หายไปด้วย mean ทุกๆ30s 

# In[7]:


#df = df.set_index('uts').resample('60S').mean()
#df


# #### Data Cleaning
#     2.2)interpolate 

# In[8]:


df = df.set_index('uts').interpolate()
df


# #### Data Cleaning 
#     3)ทำ moving average เพื่อลด noise 

# In[9]:


#mva
df = df.rolling("3s").mean()
df = df.dropna()
df.head(20)


# #### Data Normalization
#     ทำการ standardized Norm

# In[10]:


#scaler
means = df.mean()
stds = df.std()


# In[11]:


df = (df - means) / stds
df.head(20)


# ### 1.2 ขั้นตอนการแสดงข้อมูลเชิงกราฟ 

# #### แสดงกราฟข้อมูลแต่ละ feature (Column) ด้วย Line Plot เพื่อดูค่าที่แท้จริง 

# In[12]:


df.reset_index().plot(x='uts', subplots=True, figsize=(24, 36))


# #### แสดงกราฟข้อมูลความสัมพันธ์ระหว่างคู่ features ด้วย 2D Scatter Pair Plot หรือ 2D sns.jointplot หรือ 3D Scatter Plot เพื่อดูความสัมพันธ์ของข้อมูลเชิง 3 มิติ (accelerateX, accelerateY, accelerateZ) หรือ (gyro.x, gyro.y, gyro.z)

# In[13]:


fig = plt.figure(figsize=(24, 8))
ax = fig.add_subplot(1, 2, 1, projection='3d')

ax.scatter(df['accelerateX'],df['accelerateY'],df['accelerateX'],s=20,edgecolor='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylabel('z')
ax.view_init(30, -30)

ax = fig.add_subplot(1, 2, 2, projection='3d')

ax.scatter(df['gyro.x'],df['gyro.y'],df['gyro.z'],s=20,edgecolor='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_ylabel('z')
ax.view_init(30, -30)


# #### แสดงข้อมูลเชิงพิกัด Geolocation ของ ข้อมูล (gps.x, gps.y) 

# In[14]:


map_im = plt.imread('map.png')
fig, ax = plt.subplots(figsize=(14,14))
BBox = [100.2559,100.3486,13.5383,13.6124]
ax.scatter(df_copy['gps.y'], df_copy['gps.x'], zorder=1, alpha=0.5, c='r', s=10)
ax.set_title('Plotting Spatial Data on Map')
ax.set_xlim(BBox[0], BBox[1])
ax.set_ylim(BBox[2], BBox[3])
ax.imshow(map_im, zorder=0, extent=BBox, aspect='auto')


# ### 1.3 ขั้นตอนการจัดเตรียมข้อมูลเพื่อนำเข้าโมเดล 

# #### ทำการจัดข้อมูล 5 Features [accelerateX, accelerateY, accelerateZ, compass, heartrate] ในรูปของอะเรย์ 2 มิติ 2 ชุด 

# #### อะเรย์ชุดที่ 1: เป็นการจัดเรียงข้อมูล โดยต้องการ row: single time sample / column: 5 features เพื่อให้ได้ผลลัพธ์เป็นอะเรย์ขนาด  (shape: ( #sample, #features )) 

# In[15]:


df.head(50)


# In[16]:


columns = ['accelerateX', 'accelerateY', 'accelerateZ', 'compass', 'heartrate']
arr = df[columns].to_numpy()
arr.shape


# In[17]:


sns.heatmap(arr)


#  #### อะเรย์ชุดที่ 2: เป็นการจัดเรียงข้อมูล time series ในรูปของ อะเรย์ 3 มิติ โดยต้องการตัด ข้อมูลตาม time series เงื่อนไข time_step และ time stride ที่นศ.กำหนด เพื่อให้ได้ ผลลัพธ์เป็นอะเรย์ขนาด  (shape: ( #ชุด time_series, #time_step, #features ))  จากนั้นปรับอะเรย์ 3 มิติที่ได้ ให้อยู่ในรูปของ 2 มิติขนาด  (shape: ( #ชุด*#time_step, #features )) 

# In[18]:


timestep = 3
stride = 1
data = []
for i in range(0, len(df)-timestep+1, stride):
    data.append(df[columns].iloc[i: i+timestep].to_numpy())


# In[19]:


data = np.array(data)
data.shape


# In[20]:


data


# In[21]:


data = np.concatenate(data)


# In[22]:


data


# In[23]:


sns.heatmap(data)


# In[ ]:




