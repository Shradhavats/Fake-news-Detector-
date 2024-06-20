#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[4]:


data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')


# In[5]:


data_fake.head()


# In[6]:


data_true.head()


# In[7]:


data_fake["class"]=0
data_true["class"]=1


# In[8]:


data_fake.shape,data_true.shape#for the number of rows x colms


# In[9]:


#dropping the last 10 rows for testing
data_fake_manual_testing = data_fake.tail(10)
for i in range(23480,23470,-1):
    data_fake.drop([i], axis = 0, inplace = True)

data_true_manual_testing = data_true.tail(10)
for i in range(21416,21406,-1):
    data_true.drop([i], axis = 0, inplace = True)


# In[10]:


data_fake.shape, data_true.shape


# In[11]:


data_fake_manual_testing["class"]=0
data_true_manual_testing["class"]=1


# In[12]:


data_fake_manual_testing.head(10)


# In[13]:


data_true_manual_testing.head(10)


# In[14]:


data_merge = pd.concat([data_fake,data_true], axis = 0)
data_merge.head(10)


# In[15]:


data_merge.columns


# In[16]:


#removing the unnecessary colms
data = data_merge.drop(['title','subject','date'], axis = 1)


# In[17]:


data.isnull().sum()


# In[18]:


#shuffling the dataset
data = data.sample(frac = 1)


# In[19]:


data.head()


# In[20]:


data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace = True)


# In[21]:


data.columns


# In[22]:


data.head()


# In[23]:


#removing special characters
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[24]:


data['text'] = data['text'].apply(wordopt)


# In[25]:


x = data['text']
y = data['class']


# In[26]:


#splitting data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25)


# In[27]:


#converting text to vectors
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)#learning and applying tfidf
xv_test = vectorization.transform(x_test)#applying the learned tfidf on test data


# In[28]:


#for linear  regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xv_train, y_train)


# In[29]:


pred_lr = LR.predict(xv_test)


# In[30]:


#checking accuracy
LR.score(xv_test, y_test)


# In[31]:


print(classification_report(y_test, pred_lr))


# In[32]:


#for decision tree classification
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[33]:


pred_dt = DT.predict(xv_test)


# In[34]:


DT.score(xv_test, y_test)


# In[35]:


print(classification_report(y_test, pred_dt))


# In[36]:


# for Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train, y_train)


# In[37]:


pred_gb = GB.predict(xv_test)


# In[38]:


GB.score(xv_test, y_test)


# In[39]:


print(classification_report(y_test, pred_gb))


# In[40]:


from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(random_state = 0)
RF.fit(xv_train, y_train)


# In[41]:


pred_rf = RF.predict(xv_test)


# In[42]:


RF.score(xv_test, y_test)


# In[43]:


print(classification_report(y_test, pred_rf))


# In[44]:


def output_lable(n):
    if n == 0:
        return "fake News"
    elif n == 1:
        return "Not a fake news"

def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GB = GB.predict(new_xv_test)
    pred_RF = RF.predict(new_xv_test)
    
    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGB Prediction: {} \nRF Prediction: {}".format(output_lable(pred_LR[0]),
                                                                                                            output_lable(pred_DT[0]),
                                                                                                            output_lable(pred_GB[0]),
                                                                                                            output_lable(pred_RF[0])))


# In[45]:


news = str(input())
manual_testing(news)


# In[46]:


news = str(input())
manual_testing(news)


# In[ ]:





# In[ ]:




