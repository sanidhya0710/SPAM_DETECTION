#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('spam.csv', encoding='latin1')  # Assuming 'latin1' encoding


# In[3]:


df.sample(5)
df.shape


# In[4]:


df.info()


# In[5]:


# drop last three columns
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[6]:


df.sample(5)


# In[7]:


#renaming the columns
df=df.rename(columns={'v1':'target','v2':'text'})


# In[8]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target']=encoder.fit_transform(df['target'])


# In[9]:


df.head()


# In[10]:


#missing values
df.isnull().sum()


# In[11]:


#check for duplicate values
df.duplicated().sum()


# In[12]:


#remove duplicate
df=df.drop_duplicates(keep='first')


# In[13]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")


# In[14]:


import nltk


# In[15]:


nltk.download('punkt')


# In[16]:


df['num_characters']=df['text'].apply(len)


# In[17]:


df.head()


# In[18]:


#num of words
df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[19]:


df.head()


# In[20]:


df['num_sent']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[21]:


df.head()


# In[22]:


df.describe()


# In[23]:


#for ham messages 
df[df['target']==0].describe()


# In[24]:


#for spam messages
df[df['target']==1].describe()


# In[25]:


import seaborn as sns


# In[26]:


sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'],color='green')


# In[27]:


sns.pairplot(df,hue='target')


# In[28]:


df.corr()


# In[29]:


sns.heatmap(df.corr(),annot=True)


# In[30]:


#lowewr case,tokenize text,
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[40]:


transform_text('Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...')


# In[41]:


from nltk.corpus import stopwords
nltk.download('stopwords')
#stopwords.words('english')


# In[42]:


df['text'][0]


# In[43]:


import string
string.punctuation 


# In[44]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
ps.stem("rowing")


# In[45]:


df['transformed_text']=df['text'].apply(transform_text)


# In[46]:


df.head()


# In[47]:


import wordcloud


# In[48]:


from wordcloud import WordCloud
wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[49]:


spam_wc=wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))


# In[50]:


plt.imshow(spam_wc)


# In[51]:


ham_wc1=wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))


# In[52]:


plt.imshow(ham_wc1)


# In[53]:


df.head()


# In[54]:


spam_corpus=[]
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[55]:


len(spam_corpus)


# In[56]:


#words in spam with their occurance
from collections import Counter
sns.barplot(x=pd.DataFrame(Counter(spam_corpus).most_common(30))[0],y=pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[57]:


pd.DataFrame(Counter(spam_corpus).most_common(30))[1]


# In[58]:


ham_corpus=[]
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[59]:


len(ham_corpus)


# In[60]:


from collections import Counter
sns.barplot(x=pd.DataFrame(Counter(ham_corpus).most_common(30))[0],y=pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# # MODEL BUILDING

# In[150]:


from sklearn.ensemble import VotingClassifier
svc=SVC(kernel='sigmoid',gamma=1.0,probability=True)
mnb=MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50,random_state=2)
voting=VotingClassifier(estimators=[('svm',svc),('nb',mnb),('et',etc)],voting='soft')
voting.fit(X_train,y_train)
y_pred=voting.predict(X_test)
print('acc',accuracy_score(y_test,y_pred))
print('preci',precision_score(y_test,y_pred))


# In[151]:


#first we will use naive bais algorithm
#bag of words,as it is not giving as good as we wanted we will now use the tfvectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
tfidf=TfidfVectorizer(max_features=3000)


# In[152]:


#X=cv.fit_transform(df['transformed_text']).toarray()
X=tfidf.fit_transform(df['transformed_text']).toarray()


# In[153]:


#from sklearn.preprocessing import MinMaxScaler
#scaler=MinMaxScaler()
#X=scaler.fit_transform(X) scaling done
#X =np.hstack((X,df['num_characters'].values.reshape(-1,1))) adding the num_character column


# In[154]:


y =df['target'].values


# In[155]:


y


# In[156]:


from sklearn.model_selection import train_test_split


# In[157]:


X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=2)


# In[158]:


from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB


# In[159]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[160]:


gnb.fit(X_train,y_train)
y_pred1=gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[161]:


mnb.fit(X_train,y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))
#as here precision score is the must so we will go with tfidfVectorizer as it is giving precision score 1 but giving a little less accuracy as compared to cvVectorizer


# In[162]:


#bernolli is doing the best till now as the precision score is good in this case
bnb.fit(X_train,y_train)
y_pred3=bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[136]:


#tfidf->mnb


# In[137]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[138]:


svc=SVC(kernel='sigmoid',gamma=1.0)
knc=KNeighborsClassifier()
mnb=MultinomialNB()
dtc=DecisionTreeClassifier(max_depth=5)
lrc=LogisticRegression(solver='liblinear',penalty='l1')
rfc=RandomForestClassifier(n_estimators=50,random_state=2)
abc=AdaBoostClassifier(n_estimators=50,random_state=2)
bc=BaggingClassifier(n_estimators=50,random_state=2)
etc=ExtraTreesClassifier(n_estimators=50,random_state=2)
gbdt=GradientBoostingClassifier(n_estimators=50,random_state=2)


# In[139]:


clfs={
    'SVC' :svc,
    'KN' :knc,
    'NB' :mnb,
    'DT' :dtc,
    'LR' :lrc,
    'RF' :rfc,
    'ADB' :abc,
    'BC' :bc,
    'ETC' :etc,
    'GBDT' :gbdt
    
}


# In[140]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision= precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[141]:


train_classifier(mnb,X_train,y_train,X_test,y_test)


# In[100]:


accuracy_scores=[]
precision_scores=[]
for name,clf in clfs.items():
    current_accuracy,current_precision=train_classifier(clf,X_train,y_train,X_test,y_test)
    
    print('for ',name)
    print('Accuracy - ',current_accuracy)
    print('precision - ',current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

    


# In[103]:


performance_df=pd.DataFrame({'algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores})


# In[105]:


temp_df=pd.DataFrame({'algorithm':clfs.keys(),'Accuracy_max_3000':accuracy_scores,'Precision_max_3000':precision_scores})


# In[108]:


performance_df.merge(temp_df,on='algorithm')


# # model improvement

# In[ ]:


#1) change the fax_features parameter of Tfidf 
#2) use scalling in X ,we use MinMaxScaler as the naive 
#bayes cannot use negative values and standard scaler gives negative values also
#but scaling dont give good results so we will not use it
#3) now we will try to add the number of characters column that we have made previously.But it also didnt give the improvement so we will leave it
#4)now we will try voting methord as in this we can use more than 1 algorithm or models to make our accuracy and precision more accurate
#5)now we will try stacking as it is a type of voting with giving weightage to perticular models


# In[165]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))
pickle.dump(voting,open('model1.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




