# -*- coding: utf-8 -*-
"""
Created on Thu May 16 06:58:19 2019

@author: tutor
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import nltk
from nltk.corpus import PlaintextCorpusReader
import os
import re
import codecs
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
from sklearn.utils import shuffle
from flask import Flask,render_template,url_for,request

app = Flask(__name__)
global Classifier
global Vectorizer
inci=codecs.open("C:/Users/v-surkal/Desktop/DS (2)/DS/Tax_March.csv", "r",encoding='utf-8', errors='replace')
incidents = pd.read_csv(inci)
incidents["Short_Description"]=incidents["Title"]
incidents["Assignment_Group"]=incidents["Custom Category"]
incidents.drop(['Title', 'Custom Category'], axis=1, inplace=True)
incidents.dropna(inplace=True)
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 

def custom_preprocessor(text):
    text = re.sub(r'\W+|\d+|_', ' ', text)    #removing numbers and punctuations
    #  text = shortword.sub('', text)
    text = nltk.word_tokenize(text)       #tokenizing
    text = [word for word in text if not word in stop_words] #English Stopwords
    text = [lemmatizer.lemmatize(word) for word in text]              #Lemmatising
    return text


def untokenize(words):
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
############ Applying cleaning functions for shortdescription

incidents['Clean_Data']=incidents['Short_Description'].apply(custom_preprocessor)
incidents['Assignment_Group_labelled']= labelencoder.fit_transform(incidents['Assignment_Group'])
#incidents['Assignment_Group'] = incidents['Assignment_Group'].map({'Service Operations-HR Core/Self-Service': 0, 'Service Engineering-HR Core/Self-Service': 1})
incidents.drop(['Short_Description'], axis=1, inplace=True)
incidents['Desc']=incidents['Clean_Data'].apply(untokenize)
incidents.drop(['Clean_Data'], axis=1, inplace=True)
incidents.drop(['Assignment_Group'], axis=1, inplace=True)

incidents = shuffle(incidents)
train_data = incidents[:680]
test_data = incidents[680:]

# train the classifier with best accuracy
Classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(train_data.Desc)
Classifier.fit(vectorize_text, train_data.Assignment_Group_labelled)
score = Classifier.score(vectorize_text, train_data.Assignment_Group_labelled)
print(score)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
        if request.method == 'POST':
            MES = request.form['message']
            data = [MES]
            vect1 = Vectorizer.transform(data)
            my_prediction = Classifier.predict(vect1)[0]
        return render_template('result_rp.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True, use_reloader=True)
