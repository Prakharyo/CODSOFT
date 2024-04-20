#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('churn.csv')


# In[4]:


df.head()


# In[5]:


y = df.iloc[:,13]
y.head()


# In[6]:


X = df.iloc[:,3:13]
X.head()


# In[7]:


sns.countplot(y)


# In[10]:


# sns.countplot(X['Geography'],palette='pastel')


# In[11]:


labels = ['Male','Female']
sizes = X['Gender'].value_counts()
print(sizes)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
plt.show()


# In[12]:


sns.barplot(x='Geography', y='Exited', data=df)


# In[13]:


sns.barplot(x='Gender', y='Exited', data=df,palette='rocket')


# In[14]:


df.Age.plot(kind = 'hist', bins = 200, figsize = (12,12))
plt.show()


# In[15]:


g = sns.FacetGrid(df, row='Gender', col='Exited', height=4)
g.map(plt.hist,'Age', alpha=0.5, bins=20)
g.add_legend()
plt.show()


# In[16]:


g = sns.FacetGrid(df, row='IsActiveMember', col='Exited', height=4)
g.map(plt.hist,'Age', alpha=0.5, bins=20)
g.add_legend()
plt.show()


# In[17]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label = LabelEncoder()
X['Gender'] = label.fit_transform(X['Gender'])
print(X['Gender'].head(7))


# In[18]:


X['Geography']=label.fit_transform(X['Geography'])
print(X['Geography'].head())
X['Geography'].value_counts()


# In[19]:


import warnings
warnings.filterwarnings('ignore')


# In[21]:


# onehotencoding = OneHotEncoder(categorical_features  = [1])
# X = onehotencoding.fit_transform(X).toarray()
# print(X)


# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.35,random_state=42)


# In[23]:


sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


# In[24]:


X_train


# In[25]:


X_test


# In[26]:


#Shape of train and test data
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[27]:


# Using KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[28]:


#Train Model  
neigh = KNeighborsClassifier(n_neighbors = 4).fit(X_train,y_train)


# In[29]:


#Prediction
prediction = neigh.predict(X_test)


# In[30]:


prediction1=pd.DataFrame(prediction)
prediction1.head()


# In[31]:


#Accuracy
from sklearn import metrics
percent1 = metrics.accuracy_score(y_test, prediction)
percent1


# In[32]:


# Now Using SVM Algorithm 
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)


# In[33]:


#Predict
y_pred=classifier.predict(X_test)


# In[34]:


prediction2=pd.DataFrame(y_pred)
prediction2.head()


# In[35]:


#Accuracy
percent2 =metrics.accuracy_score(y_test, prediction2)
percent2


# In[36]:


from sklearn.ensemble import RandomForestClassifier
classifier_4 = RandomForestClassifier(n_estimators=100) #warning 10 to 100
classifier_4.fit(X_train,y_train)


# In[37]:


#Predict
y_randomfor=classifier_4.predict(X_test)


# In[38]:


prediction3=pd.DataFrame(y_randomfor)
prediction3.head()


# In[39]:


#Accuracy
percent3 = metrics.accuracy_score(y_test, prediction3)
percent3


# In[40]:


#Using DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
TeleTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)


# In[41]:


TeleTree.fit(X_train,y_train)


# In[42]:


y_predtree = TeleTree.predict(X_test)


# In[43]:


prediction4=pd.DataFrame(y_pred)
prediction4.head()


# In[44]:


#Accuracy
percent4 = metrics.accuracy_score(y_test,prediction4)
percent4


# In[45]:


models = pd.DataFrame({'name_model':["KNN","SVM","Random Forest","Decision Trees"],\
                        'accuracy_percentage':[percent1,percent2,percent3,percent4]})


# In[46]:


models


# In[ ]:




