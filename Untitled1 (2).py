#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston


# In[2]:


# understanding the datasets
boston = load_boston()
print(boston.DESCR)


# In[3]:


# access data attributes
dataset = boston.data
for name , index in enumerate(boston.feature_names):
    print(name,index)


# In[7]:


# reshaping data
data = dataset[:,12].reshape(-1,1)


# In[8]:


# shape of the data
np.shape(dataset)


# In[9]:


target = boston.target.reshape(-1,1)


# In[10]:


np.shape(target)


# In[11]:


# ensuring that matplotlib is working inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color='green')
plt.xlabel('Lower income population')
plt.ylabel('Cost of House')
plt.show()


# In[12]:


# regression
from sklearn.linear_model import LinearRegression

# creating a regression model
reg = LinearRegression()

# fit the model
reg.fit(data, target)


# In[13]:


# prediction
pred = reg.predict(data)


# In[14]:


# ensuring that matplotlib is working inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color='green')
plt.xlabel('Lower income population')
plt.ylabel('Cost of House')
plt.show()


# In[15]:


# ensuring that matplotlib is working inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color='red')
plt.plot(data, pred, color='green')
plt.xlabel('Lower income population')
plt.ylabel('Cost of House')
plt.show()


# In[16]:


# circumventing curve issue using polynomial model
from sklearn.preprocessing import PolynomialFeatures

# to allow merging of models
from sklearn.pipeline import make_pipeline


# In[17]:


model = make_pipeline(PolynomialFeatures(3), reg)


# In[18]:


model.fit(data, target)


# In[19]:


pred = model.predict(data)


# In[20]:


# ensuring that matplotlib is working inside the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color='red')
plt.plot(data, pred, color='green')
plt.xlabel('Lower income population')
plt.ylabel('Cost of House')
plt.show()


# In[21]:


# r_2 metric
from sklearn.metrics import r2_score


# In[22]:


# predict
r2_score(pred,target)


# In[ ]:




