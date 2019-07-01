

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 23:10:31 2019

@author: Surya Kallakuri
"""

import os
import pandas as pd
import numpy as np
from flask import Flask,render_template,url_for,request
import os
import re
import codecs

app = Flask(__name__)
global model
global tfidf

inci=codecs.open("D:/Siva/DS/Tax_March.csv", "r",encoding='utf-8', errors='replace')
df = pd.read_csv(inci)
df.columns = ['Title', 'Custom Category']
df['Custom_Category'] = df['Custom Category'].factorize()[0]
df.drop(['Custom Category'], axis=1, inplace =True)
Custom_Category_df = df[['Custom_Category']].drop_duplicates().sort_values('Custom_Category')
#Custom_Category_df = dict(Custom_Category_df.values)
#id_to_category = dict(Custom_Category_df[['Custom_Category']].values)

# remove special characters, numbers, punctuations
from io import StringIO
df["Title"] = df["Title"].str.replace("[^a-zA-Z#]", " ")
import nltk
#nltk.download()


## Stop words removal
from nltk.corpus import stopwords
stops = stopwords.words('english')

ad_stopwords = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once',
                'during', 'out', 'very', 'having', 'they', 'own', 'an', 'be', 'some', 'do', 'its', 
                'urs', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'am', 'who', 'him', 
                'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 
                'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 
                'should', 'our', 'their', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'no', 
                'when', 'at', 'before', 'them', 'same', 'been', 'have', 'will', 'on', 'does', 
                'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 
                'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 
                'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'theirs', 
                'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'] 
stops.extend(ad_stopwords)

df['Title'] = df['Title'].apply(lambda x : ' '.join([item for item in x.split() if item.lower() not in stops]))


##lemmatization
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 

df['Title'] = [lemmatizer.lemmatize(word) for word in df['Title']]

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df['Title']).toarray()
labels = df.Custom_Category
features.shape


# Splitting into training and test sets
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
##from sklearn. import train_test_split

#x = df['Title']
#y = df['Custom Category']

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20,random_state = 0)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.20, random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
#from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
    ]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
      model_name = model.__class__.__name__
      accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
      for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
        
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
#import seaborn as sns
#sns.boxplot(x='model_name', y='accuracy', data=cv_df)
#sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor='gray', linewidth=2)
#plt.show()

#cv_df.groupby('model_name').accuracy.mean()



model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
confustionMatrix = confusion_matrix(y_test, y_pred)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


#logreg_precision = confustionMatrix[1,1]/confustionMatrix[1,:].sum()
#logreg_recall = confustionMatrix[1,1]/confustionMatrix[:,1].sum()
#logreg_F1_score = (2*logreg_precision*logreg_recall)/(logreg_precision+logreg_recall)
#Accuracyscore = accuracy_score(y_test, y_pred)
#print('Accuracyscore', Accuracyscore)
#classificationreport = classification_report(y_test, y_pred)
#
#
#print('logreg_precision:',logreg_precision)
#print('logreg_recall:',logreg_recall)
#print('logreg_F1_score:',logreg_F1_score)
#print('Accuracyscore', Accuracyscore)
#print('classificationreport', classificationreport)
#print("Accuracy score", f1_score(y_test, y_pred, average='micro'))

@app.route('/')
def hello():
    return "Hello World!"
  
    
    
#SMS = 'Access to "onetaxdit.database.windows.net"'
#vectorize_message = tfidf.transform([SMS])
#predict = model.predict(vectorize_message)[0]
#print('Accuracyscore', Accuracyscore)
#score = f1_score(y_test, predict)
#print(predict)






