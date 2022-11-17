# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:40:35 2022

@author: Tobias
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

from urllib.parse import urlparse

# sklearn libraries
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split

# deployment libraries
import pickle


filename = "url_detection.pkl"


app = Flask(__name__)

def makeTokens(f):
    tokens_slash = str(f.encode('utf-8')).split('/') # make tokens after splitting by slash
    total_tokens = []
    scheme = ""
    url_parsed = urlparse(f)
    scheme = url_parsed.scheme
    for i in tokens_slash:
        # split tokens by dash character
        tokens = str(i).split('-') 
        tokens_dot = []
        for j in range(0,len(tokens)):
            # split tokens by dot
            temp_tokens = str(tokens[j]).split('.') 
            tokens_dot = tokens_dot + temp_tokens
        total_tokens = list(scheme) + total_tokens + tokens + tokens_dot
    total_tokens = list(set(total_tokens)) #remove redundant tokens
    if 'com' in total_tokens:
        total_tokens.remove('com') #removing .com since it occurs a lot of times and it should not be included in our features
    return total_tokens


model = pickle.load(open(filename,'rb'))
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))     # load vectorizer

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    url_input = []
    url_input = request.form["urlinput"]
    #final_features = [np.array(url_input)]
    #arr2 = np.reshape(final_features,(-1,1))
    #arr3 = np.array(arr2, dtype=int)
    
    vectorizer2 = TfidfVectorizer()
    #url_input = vectorizer2.fit_transform([url_input])
    #url_input = [url_input]

    #prediction = model.predict(arr2)
    #prediction = model.predict(url_input)    

    X_predict2 = vectorizer.transform([url_input])
    New_predict = model.predict(X_predict2)

    return render_template('index.html', prediction_text='URL is classified as $ {}'.format(New_predict))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)