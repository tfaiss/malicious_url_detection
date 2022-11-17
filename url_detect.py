# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:31:41 2022

@author: Tobias
"""

#import libraries
import numpy as np
from flask import Flask, render_template,request
import pickle #Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('url_detection.pkl', 'rb'))