import requests as rq
import pandas as pd
import numpy as np
import re
import pickle
import base64
import string
from collections import Counter
from nltk.corpus import stopwords
from spacy.lang.en import English
parser = English()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score
stop_list = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
symbols = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]

print('Hellowwwwwww')

# Import text processing pipeline and model 
MODEL = pickle.load(open('flaskexample/models/clf_nb.pkl', 'rb'))

# Import fitted VectorizerList: N-gram, tfidf, etc.
vectorizerList_fit = pickle.load(open('flaskexample/models/vect_fit.pkl', 'rb'))

#Define functions for text processing: cleaning, lemmatization, etc.
class CleanTextTransformer(TransformerMixin):
    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}
    
def cleanText(text):
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text

def cleanTextList(texts):
    cleanList = [] 
    for text in texts: 
    	cleanList.append(cleanText(text))
    return cleanList

def tokenizeText(sample):
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    tokens = [tok for tok in tokens if tok not in stop_list]
    tokens = [tok for tok in tokens if tok not in symbols]
    return tokens
    
def tokenizeTextList(samples):
    tokenList = []
    for sample in samples:	
    	tokenList.append(tokenizeText(sample))
    return tokenList

#Get Top10 predictive features of each trail
def get_top10_features(word, trail_features, feature_easy_list, feature_mod_list, feature_hard_list):
    final = []
    if word == 'HARD':
        top50_list = feature_hard_list
        for i, feature in enumerate(top50_list):
            if feature in trail_features:
                final.append(feature)
                if len(final) >= 13:
                    return final[3:13]
        return final
    elif word == 'MODERATE':
        top50_list = feature_mod_list
        for i, feature in enumerate(top50_list):
            if feature in trail_features:
                final.append(feature)
                if len(final) >= 12:
                    return final[0:10]
        return final
    elif word == 'EASY':
        top50_list = feature_easy_list
        for i, feature in enumerate(top50_list):
            if feature in trail_features:
                final.append(feature)
                if len(final) >= 12:
                    return final[0:10]
        return final
    else:
        return 'Whoops, something went wrong :( Refresh or pick another trail!'

#Get new prediction terms: Easy, Moderately Easy, Moderate, Moderately Hard, Hard
def new_prediction(rating):
    if rating >= 8:
        return 'Hard'
    elif rating >= 6:
        return 'Moderately Hard'
    elif rating >= 4:
        return 'Moderate'
    elif rating >= 2:
        return 'Moderately Easy'
    else:
        return 'Easy'
        
