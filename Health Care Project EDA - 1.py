#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd 
dataset = pd.read_csv(r'C:\Users\Noman\Desktop\Health Care Project/HCDS.csv')


# In[14]:


dataset.head()


# In[16]:


dataset.isna().sum()


# In[17]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm 
from IPython.core.display import display, HTML


# In[18]:


pd.set_option("display.max_columns", None )
display(HTML("<style>.container {width:100% !important; }</style>"))
plt.style.use('seaborn-whitegrid')


# In[19]:


data = pd.read_csv(r'C:\Users\Noman\Desktop\Health Care Project/HCDS.csv')


# In[20]:


data.head()


# In[21]:


data.age_in_years.mean()


# In[22]:


data.columns


# In[23]:


data.drop(columns=['state', 'previous_visit_date'], axis=1, inplace=True)


# In[24]:


data.isna().sum().sum()


# In[25]:


data.fillna(0, inplace=True)


# In[26]:


get_ipython().run_cell_magic('time', '', 'data.isna().sum().sum()')


# In[27]:


data.shape


# In[28]:


data.empi.nunique()


# In[29]:


data.empi.nunique()


# In[30]:


data.gender.value_counts()


# In[31]:


type(data['visit_start_date'][0])


# # Admission vs Readmission | Ip visit vs Ip readmission visit
# 

# In[32]:


data_temp = data.copy()


# In[33]:


data_temp['visit_start_date_M'] = pd.to_datetime(data_temp['visit_start_date']).dt.to_period('M').astype('str')
data_temp = data_temp.groupby('visit_start_date_M').aggregate({'readmission_flag':['count', 'mean', 'sum']})
print(data_temp)


# In[35]:


# - histgram = total visit 
# - linechart = readmission count 

# |_____|

fig  = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(1,1,1)
ax1.bar(x= data_temp.index, height = data_temp[("readmission_flag", "count")],  alpha= 0.8, color = 'blue')
ax1.text(s = "# of IP visit", x = -3.55, y = 400, fontsize= 20, alpha=0.7, color='black', rotation= 90)
ax1.set_xlabel("Month", fontsize=15, alpha=0.7, color='black')
ax1.set_xticklabels(data_temp.index, rotation= 45, alpha=0.7)
ax1.set_ylim(0, 1500)
plt.grid()

ax2 = ax1.twinx()
ax2.plot(data_temp.index, 100*data_temp[("readmission_flag", "mean")], alpha=0.7, color = 'darkorange', marker='o', linestyle='dashed', 
        linewidth=2, markersize=7)
ax2.text(s="IP readmission Proportion", x = 26, y = 5, fontsize=20, alpha=0.7, color='black', rotation=90)
ax2.set_yticklabels(['0%', '5%', '10%', '15%', '25%', '30%'], alpha =0.7)
ax2.set_ylim(0, 28)
plt.grid()

ax2.text(s = "      Ip Visit & Ip readmission Trend.   ", x = 4, y= 28, weight='bold', fontsize= 20, alpha=0.7)
ax1.legend(['visit_count'], loc = 'upper right', bbox_to_anchor=(0.93, 0.97))
ax2.legend(['IP readmission Trend'], loc = 'upper right', bbox_to_anchor=(0.99, 0.92))

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.show()


# Observation :
# 
# Total span of years is between Jan-18 to Dec- 19
# About 850 visits per month are recorded on an averge
# Overall average readmission proportion hovers between 13% - 14%
# Some cyclic trend is observed but not guaranteed
# Max reamdission month is repetative - may be because of year end
