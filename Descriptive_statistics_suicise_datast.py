
# coding: utf-8

# In[1]:


import pandas as pd


# In[6]:


data_k=pd.read_csv("Z://7th sem//Data analytics//master.csv")


# In[3]:


import numpy as np
from pandas import DataFrame as df
from scipy.stats import trim_mean, kurtosis
from scipy.stats.mstats import mode, gmean, hmean


# In[4]:


data_k.describe()


# In[5]:


N = 20
P = ["suicide_no","population"]
Q = [1,2,3]
values = [[998,511], [1119,620], [1300,790]]


# In[6]:


suicide = np.concatenate([np.repeat(value, N) for value in values])


# In[7]:


data = df(data = {'iv1': np.concatenate([np.array([p]*N) for p in P]*len(Q))
                  ,'iv2': np.concatenate([np.array([q]*(N*len(P))) for q in Q])
                  ,'rt': np.random.normal(suicide, scale=100.0, size=N*len(P)*len(Q))})


# In[8]:


grouped_data = data.groupby(['iv1', 'iv2'])


# In[9]:


grouped_data['rt'].describe().unstack()


# In[10]:


grouped_data['rt'].mean().reset_index()


# In[11]:


grouped_data['rt'].aggregate(np.mean).reset_index()


# In[12]:


grouped_data['rt'].apply(gmean, axis=None).reset_index()


# In[13]:


grouped_data['rt'].apply(hmean, axis=None).reset_index()


# In[14]:


trimmed_mean = grouped_data['rt'].apply(trim_mean, .5)
trimmed_mean.reset_index()


# In[15]:


grouped_data['rt'].std().reset_index()


# In[16]:


grouped_data['rt'].quantile([.25, .5, .75]).unstack()


# In[18]:


grouped_data['rt'].var().reset_index()


# In[21]:


import pandas as pd
import matplotlib.pyplot as plt


# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


data_k.describe()


# In[30]:


data_k.shape


# In[32]:


data_k.ndim


# In[33]:


data_k['suicides_no'].plot(kind='hist',bins=100)
plt.xlabel('suicides_no')


# In[34]:


plot_data=data_k[data_k['age']=='15-24 years']
plot_data=plot_data.groupby('suicides_no')['population'].sum()
plot_data.sort_values()[-10:].plot(kind='box')
plt.title("Suicides")
plt.ylabel("Young peple of 15-24 years")


# In[43]:


x=data_k.suicides_no
y=data_k.age
plt.scatter(x,y,color='k',s=10,marker="o")
plt.xlabel("suicides_no")
plt.ylabel("age")
plt.title("Age Vs suicides")
plt.show()

