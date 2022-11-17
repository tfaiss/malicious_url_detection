# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 16:42:07 2022

@author: Tobias
"""

import requests
url = 'http://localhost:5000/api'
r = requests.post(url,json={'exp':1.8,})
#r = requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6})


print(r.json())