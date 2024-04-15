#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
import re
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# In[ ]:


sms = pd.read_csv('Spam.csv', sep='\t', names=['label','message'])


# In[ ]:


sms.head()


# In[ ]:


sms.shape


# In[ ]:


sms.drop_duplicates(inplace=True)


# In[ ]:


sms.reset_index(drop=True, inplace=True)


# In[ ]:


sms.shape


# In[ ]:


sms['label'].value_counts()


# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(x='label', data=sms)
plt.xlabel('SMS Classification')
plt.ylabel('Count')
plt.show()


# In[ ]:


corpus = []
ps = PorterStemmer()

for i in range(0,sms.shape[0]):
    message = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sms.message[i]) #Cleaning special character from the message
    message = message.lower() #Converting the entire message into lower case
    words = message.split() # Tokenizing the review by words
    words = [word for word in words if word not in set(stopwords.words('english'))] #Removing the stop words
    words = [ps.stem(word) for word in words] #Stemming the words
    message = ' '.join(words) #Joining the stemmed words
    corpus.append(message) #Building a corpus of messages


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[ ]:


best_accuracy = 0.0
alpha_val = 0.0
for i in np.arange(0.0,1.1,0.1):
    temp_classifier = MultinomialNB(alpha=i)
    temp_classifier.fit(X_train, y_train)
    temp_y_pred = temp_classifier.predict(X_test)
    score = accuracy_score(y_test, temp_y_pred)
    print("Accuracy score for alpha={} is: {}%".format(round(i,1), round(score*100,2)))
    if score>best_accuracy:
        best_accuracy = score
        alpha_val = i
print('--------------------------------------------')
print('The best accuracy is {}% with alpha value as {}'.format(round(best_accuracy*100, 2), round(alpha_val,1)))


# In[ ]:


classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


acc_s = accuracy_score(y_test, y_pred)*100


# In[ ]:


print("Accuracy Score {} %".format(round(acc_s,2)))


# In[ ]:


def predict_spam(sample_message):
    sample_message = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_message)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_message = [ps.stem(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    temp = cv.transform([final_message]).toarray()
    return classifier.predict(temp)


# In[ ]:


result = ['This is a SPAM!','Ohhh, this is a normal message.']


# In[ ]:


msg = "Hi! You are pre-qulified for Premium Axis Bank Credit Card. Also get Rs.1000 worth Myntra Gift Card*, 20X Rewards Point* & more. Click "

if predict_spam(msg):
    print(result[0])
else:
    print(result[1])


# In[ ]:


msg = "[Update] Congratulations John Mukesh, You account is activated for investment in Stocks. Click to invest now: "

if predict_spam(msg):
    print(result[0])
else:
    print(result[1])


# In[ ]:


msg = "Your Stock broker Dhikana BROKING LIMITED reported your fund balance Rs.2500.05 & securities balance 0.0 as on end of JUNE-25 . Balances do not cover your bank, DP & PMS balance with broking entity. Check details at YOGESHNILE.WORK4U@GMAIL.COM. If email Id not correct, kindly update with your broker."

if predict_spam(msg):
    print(result[0])
else:
    print(result[1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




