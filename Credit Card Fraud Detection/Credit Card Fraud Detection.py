#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install prettytable  


# In[3]:


import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
 

# from prettytable import PrettyTable
from prettytable import PrettyTable

import warnings
warnings.filterwarnings("ignore")


# In[4]:


data = pd.read_csv('CC.csv')


# In[5]:


data.head(5)


# In[6]:


data.shape


# In[7]:


data.describe()


# In[8]:


data.info()


# In[9]:


data.isnull().sum()


# In[10]:


data['Class'].value_counts()


# In[11]:


sc = StandardScaler()
amount = data['Amount'].values
data['Amount'] = sc.fit_transform(amount.reshape(-1, 1))


# In[12]:


data.head()


# In[13]:


x = data.drop('Class', axis = 1)
y = data['Class']


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42,shuffle=True)


# In[15]:


print("The shape of X_train is:",X_train.shape)
print("The shape of X_test is:",X_test.shape)
print("The shape of y_train is:",y_train.shape)
print("The shape of y_test is:",y_test.shape)


# In[16]:


logisticRegression = LogisticRegression()
logisticRegression.fit(X_train,y_train)
logisticRegression_pred = logisticRegression.predict(X_test)
logisticRegression_acc = accuracy_score(y_test, logisticRegression_pred)


# In[17]:


svc = SVC()
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
svc_acc = accuracy_score(y_test, svc_pred)


# In[18]:


# randomForestCls = RandomForestClassifier()
# randomForestCls.fit(X_train, y_train)
# randomForestCls_pred = randomForestCls.predict(X_test)
# randomForestCls_acc = accuracy_score(y_test ,randomForestCls_pred)


# In[19]:


Table = PrettyTable(["Algorithm", "Accuracy"])
Table.add_row(["LogisticRegression", logisticRegression_acc])
Table.add_row(["SVC", svc_acc])
# Table.add_row(["RandomForestClassifier", randomForestCls_acc])
print(Table)


# In[ ]:





# In[ ]:




