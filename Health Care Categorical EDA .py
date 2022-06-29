#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from tqdm import tqdm 
from IPython.core.display import display, HTML


# In[2]:


dataset = pd.read_csv(r'C:\Users\Noman\Desktop\Health Care Project/HCDS.csv')


# In[3]:


pd.set_option("display.max_columns", None )
display(HTML("<style>.container {width:100% !important; }</style>"))
plt.style.use('seaborn-whitegrid')


# In[4]:


data = pd.read_csv(r'C:\Users\Noman\Desktop\Health Care Project/HCDS.csv')


# In[5]:


data.head()


# In[6]:


def labeled_barplot(data, feature, prec=False, n=None):
    total = len(data[feature])
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count +1, 5))
    else:
        plt.figure(figsize=(n+1, 5))
    
    plt.xticks(rotation=90,fontsize=15)
    ax = sns.countplot(
    data=data, x=feature, palette="Paired", order=data[feature].value_counts().index[:n].sort_values(),)
    for p in ax.patches:
        if prec == True:
            label = "{:.1f}%".format(
            100 * p.get_height()/total)
        else:
            label = p.get_height()
    x = p.get_x() + p.get_width()
    y = p.get_height()
    
    ax.annotate(
    label, (x,y), ha="center", va="center", size=12, xytext=(0,5), textcoords = "offset points",)
    plt.show()


# In[7]:


labeled_barplot(data, "unplanned_flag", prec=True)


# In[8]:


labeled_barplot(data, "ddc_category", prec=True)


# In[9]:


labeled_barplot(data, "Renal Disease", prec=True)


# In[10]:


data_temp = data[data.ddc_category != 'expired'].copy()
data_temp = data_temp.groupby('ddc_category').aggregate({'readmission_flag':['mean', 'count', 'sum']})
data_temp = data_temp.sort_values(by = ('readmission_flag', 'count'), ascending=False).copy()
data_temp.index = [i.capitalize() for i in data_temp.index]

fig  = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(1,1,1)
ax1.text(s = "Discharge Status vs Readmission PERC", x =-0.5, y = 17500, weight= 'bold', alpha=0.7, color='black', fontsize= 20)
ax1.bar(x = data_temp.index, height = data_temp[("readmission_flag", "count")], alpha=0.8, color='orange')
ax1.bar(x = data_temp.index, height = data_temp[("readmission_flag", "sum")], alpha=0.8, color='darkviolet')

ax1.text(s = "# Ip Visit", x =-1, y = 6000, alpha=0.7, color='black', fontsize= 15, rotation=90)
ax1.set_xlabel("Discharge Status", fontsize=15, alpha=0.7, color='black')
ax1.set_xticklabels(data_temp.index, rotation=90, alpha=0.7, fontsize=15)

rate = round(100*data_temp[('readmission_flag', 'mean')],1)
ax1.text(s = str(rate[0])+'%', x = -0.2, y=2500, weight = 'bold', fontsize=20, alpha=0.7)
ax1.text(s = str(rate[1])+'%', x = 0.8, y=2500, weight = 'bold', fontsize=20, alpha=0.7)
ax1.text(s = str(rate[2])+'%', x = 1.8, y=2500, weight = 'bold', fontsize=20, alpha=0.7)
plt.grid()
plt.legend(['# visit', "#readmission"])

plt.show()


# In[11]:


def Stacked_barplot(data, predictor, target):
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins = True).sort_values(by = sorter, ascending=False)
    print(tab1)
    print("__"*120)
    tab = pd.crosstab(data[predictor], data[target], normalize = "index").sort_values(by = sorter, ascending=False)
    tab.plot(kind='bar', stacked=True, figsize=(count+1, 5))
    plt.legend(
    loc = "lower left",
    frameon = False)
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))
    plt.show()


# In[12]:


Stacked_barplot(data, "ddc_category", "readmission_flag")


# In[ ]:




