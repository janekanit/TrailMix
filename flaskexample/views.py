from .trailmix_func import CleanTextTransformer, cleanTextList, tokenizeText, tokenizeTextList, new_prediction, get_top10_features
from flask import Flask, request, render_template
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2

import string
import requests as rq
import pandas as pd
import numpy as np
from collections import Counter
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


# Python code to connect to Postgres
user = 'jane'       
host = 'localhost'
dbname = 'trailmix2'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user, host ='localhost', password='1234')

@app.route('/')
def trail_input():  
    return render_template("input.html")


@app.route('/output')
def get_difficulty_prediction():

	#Insert model
    clf = pickle.load(open('flaskexample/models/clf_nb.pkl', 'rb'))     
    
    #Get trail name from trail_name in html
    name = request.args.get('trail_name')        

    
    #Query review texts when enter trail name
    text_query = """                                                             
               SELECT text, name FROM trailmix_data_table WHERE name = '%s' LIMIT 300\
;                                                                               
               """ % name



	#Query review texts when enter trail name for sample reviews
    text_table_query = """                                                             
               SELECT text, user FROM trailmix_data_table WHERE name = '%s' LIMIT 10\
;                                                                               
               """ % name
	
	#Read query text TABLE results	
    text_table_results = pd.read_sql_query(text_table_query, con)
    text_table_list = text_table_results['text'].values.tolist()
	
	
	#Read query text results	
    text_results = pd.read_sql_query(text_query, con)
    text_col_list = text_results['text'].values.tolist()
	
    #Clean query text results in columns
    text_list = cleanTextList(text_col_list) 
    
    #Token text results in columns and join to string 
    text_token = tokenizeTextList(text_list)
    text_string = [' '.join(x) for x in text_token]
    
    #Vectorizer from all unigram, bigram tokens
    vectorizerList_fit = pickle.load(open('flaskexample/models/vect_fit.pkl', 'rb'))
    vect_token = vectorizerList_fit.transform(text_string)
    vect_array = vect_token.toarray()
    
    #the_result: Difficulty prediction (EASY, MODERATE, HARD) from all reviews 
    MODEL = pickle.load(open('flaskexample/models/clf_nb.pkl', 'rb'))
    prediction = MODEL.predict(vect_array)
    prediction_count = Counter(prediction.flat).most_common(1)
    prediction_one = prediction_count[0]
    prediction_word = prediction_one[0]
    
    #the_rating: Weight average difficulty score (0-10) of the trail
    df_prediction = pd.Series(prediction)
    rating_dict = {'HARD':10, 'MODERATE':5,'EASY':0}
    df_prediction = df_prediction.map(rating_dict)
    prediction_mean = int(df_prediction.mean())
    
    #prediction_new: New difficulty terms based prediction_mean
    prediction_new = new_prediction(prediction_mean)
    
    #Get vectors of the trail as list: vect_trail_features
    vect_trail = CountVectorizer(stop_words='english', ngram_range=(1,2))
    vect_trail.fit_transform(text_string)
    vect_trail_features = vect_trail.get_feature_names()
    
    #Display top 20 tags/features
    #Query features
    feature_query = """                                                             
               SELECT easy_tags, mod_tags, hard_tags FROM feature2_data_table\
;                                                                               
               """
            
	#Read query features results
    feature_results = pd.read_sql_query(feature_query, con)
    easy_idx = [0]
    easy_idx.extend(range(12,30))
    feature_easy_list = feature_results['easy_tags'][easy_idx].values.tolist()
    feature_mod_list = feature_results['mod_tags'][5:30].values.tolist()
    feature_hard_list = feature_results['hard_tags'][9:30].values.tolist()
    #top_features = get_features(prediction_word, feature_easy_list, feature_mod_list, feature_hard_list)
    top_features = get_top10_features(prediction_word, vect_trail_features, feature_easy_list, feature_mod_list, feature_hard_list)
    feature1 = top_features[0]
    feature2 = top_features[1]
    feature3 = top_features[2]
    feature4 = top_features[3]
    feature5 = top_features[4]
    feature6 = top_features[5] 
    feature7 = top_features[6]
    feature8 = top_features[7]
    feature9 = top_features[8]   
    feature10 = top_features[9]                          
    
    return render_template("output.html", trail_name = name, the_result = prediction_new, the_rating = prediction_mean, reviews = text_table_list, the_features_1 = feature1, the_features_2 = feature2, the_features_3 = feature3, the_features_4 = feature4, the_features_5 = feature5, the_features_6 = feature6, the_features_7 = feature7, the_features_8 = feature8, the_features_9 = feature9, the_features_10 = feature10)

if __name__ == '__main__':
    app.run(debug=True)

    
  
