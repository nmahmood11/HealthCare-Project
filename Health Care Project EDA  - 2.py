#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm 
from IPython.core.display import display, HTML


# In[3]:


dataset = pd.read_csv(r'C:\Users\Noman\Desktop\Health Care Project/HCDS.csv')


# In[4]:


pd.set_option("display.max_columns", None )
display(HTML("<style>.container {width:100% !important; }</style>"))
plt.style.use('seaborn-whitegrid')


# In[5]:


dataset.head()


# In[7]:


data = pd.read_csv(r'C:\Users\Noman\Desktop\Health Care Project/HCDS.csv')


# In[ ]:





# In[8]:


data.columns


# In[9]:


data.unplanned_flag.value_counts()


# In[10]:


data.readmission_flag.value_counts()


# # Planned Readmission vs unplanned Readmission | Planned visit vs unplanned visit

# In[11]:


fig = plt.figure(figsize=(18, 7))
ax1 = fig.add_subplot(1,2,1)
plt.title("% Readmissions ")
params = {'kind':'pie', 'labels': ['Admissions +\n Planned Readmissions', 'Unplanned Readmissions'], 'explode':(0,0.2), 
         'autopct':'%1.1f%%'}
data.readmission_flag.value_counts().rename("").plot(**params)

ax2 = fig.add_subplot(1,2,2)
params = {'kind': 'pie', 'labels': ['Unplanned Visits', 'Planned Visits'], 'explode':(0,0.1), 
         'autopct':'%1.1f%%'}
data.unplanned_flag.value_counts().rename("").plot(**params)
plt.title("Unplanned Visits")

plt.show()


# Observation: 1. About 13% of the total visits recorded as Unplanned readmission (remaining are either planned readmission or Admission ) 2. 77.5% which is 3/4th of the total visit are unplanned in nature

# # Patient Demographics

# In[12]:


data.gender.value_counts()


# In[13]:


data.race.value_counts()


# In[14]:


fig = plt.figure(figsize=(14, 18))

ax1 = fig.add_subplot(1,2,1)
params = {'kind':'pie', 'labels': ['Female', 'Male'], 'explode':(0,0.05), 
         'autopct':'%1.1f%%'}
data.gender.value_counts().rename("").plot(**params)
plt.title("Gender ")

ax2 = fig.add_subplot(1,2,2)
params = {'kind':'pie', 'explode':(0,0.1, 0.1), 
         'autopct':'%1.1f%%'}
data.race.value_counts().rename("").plot(**params)
plt.title("Race")
plt.show()


# # What is the readmission proportion in demographics ?

# In[15]:


round(100*data.groupby('race').agg({'readmission_flag':'mean'}), 2).rename(columns = {'readmission_flag' : 'Readmission Proportion %'})


# In[16]:


round(100*data.groupby('gender').agg({'readmission_flag':'mean'}), 2).rename(columns = {'readmission_flag' : 'Readmission Proportion %'})


# Observation: 1. Almost 57% of the total visits are made of female patients 2. white patient forms the majority of the visit but less in readmission proportion

# In[17]:


data.visit_amount.mean()


# In[18]:


data[data['readmission_flag'] == 1]['visit_amount'].mean()


# In[19]:


data['taxonomy_cat'].unique()


# In[20]:


data['ddc_category'].unique()


# In[21]:


data[data['readmission_flag'] == 1]['taxonomy_cat'].value_counts()


# In[22]:


round(100*data.groupby(['taxonomy_cat']).agg({'visit_amount': 'mean', 'readmission_flag': 'mean'}),                                         2)


# # Univariate Outlier Analysis

# In[24]:


def outlier_univariate_analysis(data, feature, figsize=(12,7), kde=False, bins=None):
    f2, (ax_box2, ax_hist2) = plt.subplots(nrows=2, sharex = True, gridspec_kw = {"height_ratios": (0.25, 0.75)}, figsize=figsize)
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color='violet')
    sns.histplot(data=data, x = feature, kde=kde, ax=ax_hist2, bins=bins, palette='winter') if bins else sns.histplot(
    data=data, x=feature, kde=kde, ax=ax_hist2)
    ax_hist2.axvline(
    data[feature].mean(), color="green", linestyle="--")
    ax_hist2.axvline(
    data[feature].median(), color="black", linestyle="-")


# In[25]:


outlier_univariate_analysis(data, 'visit_amount')


# In[26]:


outlier_univariate_analysis(data, 'age_in_years')


# In[27]:


data['age_in_years'].describe().T


# In[28]:


outlier_univariate_analysis(data, 'length_of_stay')


# In[ ]:




