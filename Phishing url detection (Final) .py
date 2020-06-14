#!/usr/bin/env python
# coding: utf-8

# # Importing the required packages
/lalitha
# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
import seaborn as sns
from urllib.parse import urlparse
import tldextract
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import learning_curve


# # Loading the Train Dataset

# In[2]:


df=pd.read_csv("url train dataset.csv",delimiter = ',',encoding = 'unicode_escape', low_memory = False)


# In[102]:


df.head()


# In[103]:


df.info()


# In[104]:


df.describe()


# # Checking dataset distribution

# In[105]:


df['label'].value_counts()


# # Plot of unbalanced distribution

# In[106]:


df_ublabel0 = df[df.label==0]
df_ublabel1 = df[df.label==1]
df_ubclass = pd.concat([df_ublabel0,df_ublabel1]) 


# In[107]:


plt.figure(figsize=(5, 8))
sns.countplot('label', data=df_ubclass)
plt.title('unbalanced Classes')
plt.show()


# # Resampling and balancing dataset

# In[108]:


df_blabel1_resampled = resample(df_ublabel1, replace= False, n_samples=6251)
df_bsampled = pd.concat([df_blabel1_resampled,df_ublabel0]) 
df_bsampled.label.value_counts()                                


# In[109]:


plt.figure(figsize=(5,8))
sns.countplot('label', data=df_bsampled)
plt.title('Balanced Classes')
plt.show()


# # Creating Features from train data

# In[3]:


# Method to count number of dots
def countdots(url):  
    return url.count('.')


# In[4]:


# Method to count number of delimeters
def countdelim(url):
    count = 0
    delim=[';','_','?','=','&']
    for each in url:
        if each in delim:
            count = count + 1
    
    return count


# In[5]:


# Is IP addr present as the hostname, let's validate

import ipaddress as ip #works only in python 3

def isip(url):
    try:
        if ip.ip_address(url):
            return 1
    except:
        return 0


# In[6]:


#method to check the presence of hyphens

def isPresentHyphen(url):
    return url.count('-')


# In[7]:


#method to check the presence of @

def isPresentAt(url):
    return url.count('@')


# In[8]:


def isPresentDSlash(url):
    return url.count('//')


# In[9]:


def countSubDir(url):
    return url.count('/')


# In[10]:


def get_ext(url):
      
    root, ext = splitext(url)
    return ext


# In[11]:


def countSubDomain(subdomain):
    if not subdomain:
        return 0
    else:
        return len(subdomain.split('.'))


# In[12]:


def countQueries(query):
    if not query:
        return 0
    else:
        return len(query.split('&'))


# In[13]:


featureSet = pd.DataFrame(columns=('url','no of dots','presence of hyphen','len of url','presence of at','presence of double slash','no of subdir','no of subdomain','len of domain','no of queries','is IP','label'))


# In[22]:


get_ipython().system('pip install tldextract')


# In[14]:



def getFeatures(url, label): 
    result = []
    url = str(url)
    
    #add the url to feature set
    result.append(url)
    
    #parse the URL and extract the domain information
    path = urlparse(url)
    ext = tldextract.extract(url)
    
    #counting number of dots in subdomain    
    result.append(countdots(ext.subdomain))
    
    #checking hyphen in domain   
    result.append(isPresentHyphen(path.netloc))
    
    #length of URL    
    result.append(len(url))
    
    #checking @ in the url    
    result.append(isPresentAt(path.netloc))
    
    #checking presence of double slash    
    result.append(isPresentDSlash(path.path))
    #Count number of subdir    
    result.append(countSubDir(path.path))
    
    #number of sub domain    
    result.append(countSubDomain(ext.subdomain))
    
    #length of domain name    
    result.append(len(path.netloc))
    
    #count number of queries    
    result.append(len(path.query))
    
    #Adding domain information
    
    #if IP address is being used as a URL     
    result.append(isip(ext.domain))
    #result.append(get_ext(path.path))
    result.append(str(label))
    return result


# In[15]:


for i in range(len(df)):
    features = getFeatures(df["domain"].loc[i],df["label"].loc[i])    
    featureSet.loc[i] = features


# In[123]:


featureSet.head()


# In[124]:


featureSet.info()


# # Visualizing of Features
# 

# In[16]:


sns.set(style="darkgrid")
sns.distplot(featureSet[featureSet['label']=='0']['len of url'],color='green',label='Benign')
sns.distplot(featureSet[featureSet['label']=='1']['len of url'],color='red',label='Phishing')
plt.title('Distribution of URL Length')
plt.legend(loc='upper right')
plt.xlabel('Length of URL')
plt.show()


# In[126]:


sns.set(style="darkgrid")
sns.distplot(featureSet[featureSet['label']=='0']['no of dots'],color='green',label='Benign')
sns.distplot(featureSet[featureSet['label']=='1']['no of dots'],color='red',label='Phishing')
plt.title('Distribution of Dots')
plt.legend(loc='upper right')
plt.xlabel('No of Dots')
plt.show()


# In[127]:


sns.set(style="darkgrid")
sns.distplot(featureSet[featureSet['label']=='0']['no of subdir'],color='green',label='Benign')
sns.distplot(featureSet[featureSet['label']=='1']['no of subdir'],color='red',label='Phishing')
plt.title('Distribution of Subdirectory')
plt.legend(loc='upper right')
plt.xlabel('No of Subdirectory')
plt.show()


# In[128]:


sns.set(style="darkgrid")
sns.distplot(featureSet[featureSet['label']=='0']['no of subdomain'],color='green',label='Benign')
sns.distplot(featureSet[featureSet['label']=='1']['no of subdomain'],color='red',label='Phishing')
plt.title('Distribution of Subdomain')
plt.legend(loc='upper right')
plt.xlabel('No of Subdomains')
plt.show()


# In[129]:


sns.set(style="darkgrid")
sns.distplot(featureSet[featureSet['label']=='0']['no of queries'],color='green',label='Benign')
sns.distplot(featureSet[featureSet['label']=='1']['no of queries'],color='red',label='Phishing')
plt.title('Distribution of Queries')
plt.legend(loc='upper right')
plt.xlabel('No of Queries')
plt.show()


# # Splitting the dataset to test and train

# In[130]:


X=featureSet.iloc[:,1:11].values
Y=featureSet.iloc[:,11].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[131]:


print(X_train.shape)
print(Y_train.shape)


# In[132]:


print(X_test.shape)
print(Y_test.shape)


# # Training classifiers and testing without hyper parameter tuning

# In[133]:


print('Decision Tree Classifier')
clf1 = tree.DecisionTreeClassifier()
clf1.fit(X_train,Y_train)
Y_pred = clf1.predict(X_test)
print('Accuraccy : %f' % metrics.accuracy_score(Y_test, Y_pred))


# In[134]:


print('Adaboost Classifier')
clf2 = AdaBoostClassifier()
clf2.fit(X_train,Y_train)
Y_pred = clf2.predict(X_test)
print('Accuraccy : %f' % metrics.accuracy_score(Y_test, Y_pred))


# In[135]:


print('Logistic Regression Classifier')
clf3 = LogisticRegression()
clf3.fit(X_train,Y_train)
Y_pred = clf3.predict(X_test)
print('Accuraccy : %f' % metrics.accuracy_score(Y_test, Y_pred))


# In[136]:


print('Gaussian NB Classifier')
clf4 = GaussianNB()
clf4.fit(X_train,Y_train)
Y_pred = clf4.predict(X_test)
print('Accuraccy : %f' % metrics.accuracy_score(Y_test, Y_pred))


# In[137]:


print('KNN Classifier')
clf5 = KNeighborsClassifier()
clf5.fit(X_train,Y_train)
Y_pred = clf5.predict(X_test)
print('Accuraccy : %f' % metrics.accuracy_score(Y_test, Y_pred))


# # Hyper parameter tuning using GridSearch CV

# In[138]:


print('Decision Tree Classifier')
param_grid = {"criterion" : ["gini", "entropy"],
              "max_depth": [3,5,20,30],
              "splitter" : ["best","random"]
             }
griddt = GridSearchCV(estimator=clf1, param_grid=param_grid)
griddt.fit(X_train,Y_train)
print(griddt)
# summarize the results of the grid search
print(griddt.best_score_)
print(griddt.best_estimator_.max_depth)
print(griddt.best_estimator_.criterion)
print(griddt.best_estimator_.splitter)


# In[139]:


print('Adaboost Classifier')
param_grid = {"learning_rate" : [1,2,3,5,6],
              "n_estimators": [5,10,15,25,50],
              }
griddt = GridSearchCV(estimator=clf2, param_grid=param_grid)
griddt.fit(X_train,Y_train)
print(griddt)
# summarize the results of the grid search
print(griddt.best_score_)
print(griddt.best_estimator_.learning_rate)
print(griddt.best_estimator_.n_estimators)


# In[140]:


print('Logistic Regression Classifier')
param_grid = {"penalty" : ["l1","l2"],
              "C": [0.1,0.5,1,1.5],
              }
griddt = GridSearchCV(estimator=clf3, param_grid=param_grid)
griddt.fit(X_train,Y_train)
print(griddt)
# summarize the results of the grid search
print(griddt.best_score_)
print(griddt.best_estimator_.penalty)
print(griddt.best_estimator_.C)


# In[141]:


print('KNN Classifier')
param_grid = {"n_neighbors" : [1,2,3,4,5],
              "weights": ["uniform","distance"],
              }
griddt = GridSearchCV(cv=5,estimator=clf5, param_grid=param_grid)
griddt.fit(X_train,Y_train)
print(griddt)
# summarize the results of the grid search
print(griddt.best_score_)
print(griddt.best_estimator_.n_neighbors)
print(griddt.best_estimator_.weights)


# # Training and validating the classifiers with tuned hyper parameters using learning curves

# In[142]:


print('Decision Tree Classifier')
clf1 = tree.DecisionTreeClassifier(max_depth =30 , criterion = 'gini', splitter = 'random')
clf1.fit(X_train,Y_train)
scores_1 = cross_val_score(clf1, X_train, Y_train, cv=5)
print("Accuracy: %f (+/- %0.2f)" % (scores_1.mean(), scores_1.std() * 2))
train_sizes, train_scores,validation_scores = learning_curve(clf1,X_train,Y_train,cv=5,scoring='accuracy', n_jobs=-1,train_sizes=np.linspace(0.01, 1.0, 50)
)
train_mean = np.mean(1-train_scores, axis=1)
validation_mean = np.mean(1-validation_scores, axis=1)
plt.plot(train_sizes, validation_mean, label = 'Validation error')
plt.plot(train_sizes, train_mean, label = 'training error')
plt.legend()


# In[143]:


print('Adaboost Classifier')
clf2 = AdaBoostClassifier(learning_rate = 1, n_estimators = 50)
clf2.fit(X_train,Y_train)
scores_2 = cross_val_score(clf2, X_train, Y_train, cv=5)
print("Accuracy: %f (+/- %0.2f)" % (scores_2.mean(), scores_2.std() * 2))
train_sizes, train_scores,validation_scores = learning_curve(clf2,X_train,Y_train,cv=5,scoring='accuracy', n_jobs=-1,train_sizes=np.linspace(0.01, 1.0, 50)
)
train_mean = np.mean(1-train_scores, axis=1)
validation_mean = np.mean(1-validation_scores, axis=1)
plt.plot(train_sizes, train_mean, label = 'training error')
plt.plot(train_sizes, validation_mean, label = 'Validation error')
plt.legend()


# In[144]:


print('Logistic Regression Classifier')
clf3 = LogisticRegression(penalty = 'l1', C = 1.5)
clf3.fit(X_train,Y_train)
scores_3 = cross_val_score(clf3, X_train, Y_train, cv=5)
print("Accuracy: %f (+/- %0.2f)" % (scores_3.mean(), scores_3.std() * 2))
train_sizes, train_scores,validation_scores=learning_curve(clf3,X_train,Y_train,cv=5,scoring='accuracy', n_jobs=-1,train_sizes=np.linspace(0.01, 1.0, 50))
train_mean = np.mean(1-train_scores, axis=1)
validation_mean = np.mean(1-validation_scores, axis=1)
plt.plot(train_sizes, train_mean, label = 'training error')
plt.plot(train_sizes, validation_mean, label = 'Validation error')
plt.legend()


# In[145]:


print('Gaussian NB Classifier')
clf4 = GaussianNB()
clf4.fit(X_train,Y_train)
scores_4 = cross_val_score(clf4, X_train, Y_train, cv=5)
print("Accuracy: %f (+/- %0.2f)" % (scores_4.mean(), scores_4.std() * 2))
train_sizes, train_scores,validation_scores = learning_curve(clf4,X_train,Y_train,cv=5,scoring='accuracy', n_jobs=-1,train_sizes=np.linspace(0.01, 1.0, 50)
)
train_mean = np.mean(1-train_scores, axis=1)
validation_mean = np.mean(1-validation_scores, axis=1)
plt.plot(train_sizes, train_mean, label = 'training error')
plt.plot(train_sizes, validation_mean, label = 'Validation error')
plt.legend()


# In[146]:


print('KNN Classifier')
clf5 = KNeighborsClassifier(n_neighbors = 4, weights = 'distance')
clf5.fit(X_train,Y_train)
scores_5 = cross_val_score(clf5, X_train, Y_train, cv=5)
print("Accuracy: %f (+/- %0.2f)" % (scores_5.mean(), scores_5.std() * 2))
train_sizes, train_scores,validation_scores = learning_curve(clf5,X_train,Y_train,cv=5,scoring='accuracy', n_jobs=-1,train_sizes=np.linspace(0.01, 1.0, 50)
)
train_mean = np.mean(1-train_scores, axis=1)
validation_mean = np.mean(1-validation_scores, axis=1)
plt.plot(train_sizes, train_mean, label = 'training error')
plt.plot(train_sizes, validation_mean, label = 'Validation error')
plt.legend()


# # Boxplot of error vs classifiers

# In[147]:


error = [1-scores_1,1-scores_2,1-scores_3,1-scores_4,1-scores_5]
plt.figure(figsize=(10,10 ))
sns.boxplot(data = error)
plt.xticks([0,1,2,3,4], ['DT','Ada Boost', 'Logistic Reg', 'Gaussian NB', 'KNN'])


# # Reducing the variance of best classifier (DT)

# In[148]:


print('Decision Tree Classifier')
clf1 = tree.DecisionTreeClassifier(max_depth =7, criterion = 'gini', splitter = 'best')
clf1.fit(X_train,Y_train)
scores_1 = cross_val_score(clf1, X_train, Y_train, cv=10)
print("Accuracy: %f (+/- %0.2f)" % (scores_1.mean(), scores_1.std() * 2))
train_sizes, train_scores,validation_scores = learning_curve(clf1,X_train,Y_train,cv=5,scoring='accuracy', n_jobs=-1,train_sizes=np.linspace(0.01, 1.0, 50)
)
train_mean = np.mean(1-train_scores, axis=1)
validation_mean = np.mean(1-validation_scores, axis=1)
plt.plot(train_sizes, validation_mean, label = 'Validation error')
plt.plot(train_sizes, train_mean, label = 'training error')
plt.legend()


# # Testing the best classifier (DT)

# In[149]:


print('Decision Tree Classifier')
Y_pred = clf1.predict(X_test)
from sklearn import metrics
print('Accuraccy : %f' % metrics.accuracy_score(Y_test, Y_pred))
cm = confusion_matrix(Y_test,Y_pred)
cr = classification_report(Y_test,Y_pred)
sns.heatmap(cm,annot=True,cbar=True,xticklabels='auto',yticklabels='auto')
print(cr)


# # Transfer learning of best classifier on new dataset using DT

# In[150]:


dftl=pd.read_csv("url transfer dataset.csv",delimiter = ',',encoding = 'unicode_escape', low_memory = False)


# In[151]:


dftl.describe()


# In[152]:


featureSettl = pd.DataFrame(columns=('url','no of dots','presence of hyphen','len of url','presence of at','presence of double slash','no of subdir','no of subdomain','len of domain','no of queries','is IP','label'))


# In[153]:


for i in range(len(dftl)):
    featurestl = getFeatures(dftl["domain"].loc[i], dftl["label"].loc[i])    
    featureSettl.loc[i] = featurestl


# In[154]:


featureSettl.info()


# In[155]:


featureSettl.head()


# In[156]:


X_testtl=featureSettl.iloc[:,1:11].values
Y_testtl=featureSettl.iloc[:,11].values
print(X_testtl.shape)
print(Y_testtl.shape)


# In[157]:


print('Decision Tree Classifier')
Y_predtl = clf1.predict(X_testtl)
print('Accuraccy : %f' % metrics.accuracy_score(Y_testtl, Y_predtl))
cm = confusion_matrix(Y_testtl, Y_predtl)
cr = classification_report(Y_testtl, Y_predtl)
sns.heatmap(cm,annot=True,cbar=True,xticklabels='auto',yticklabels='auto')
print(cr)


# # Demo using DT

# In[158]:


result = pd.DataFrame(columns=('url','no of dots','presence of hyphen','len of url','presence of at','presence of double slash','no of subdir','no of subdomain','len of domain','no of queries','is IP','label'))
results = getFeatures('https://www.google.com/', '')
result.loc[0] = results
result = result.drop(['url','label'],axis=1).values
print(clf1.predict(result))


# In[159]:


result = pd.DataFrame(columns=('url','no of dots','presence of hyphen','len of url','presence of at','presence of double slash','no of subdir','no of subdomain','len of domain','no of queries','is IP','label'))
results = getFeatures('http://12.34.56.78/firstgenericbank/account-update/', '')
result.loc[0] = results
result = result.drop(['url','label'],axis=1).values
print(clf1.predict(result))

