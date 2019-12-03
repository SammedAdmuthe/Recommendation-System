# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 22:07:11 2019

@author: samme
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 


def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
    
df = pd.read_csv("movie_dataset.csv")
"""
print(df.head())
text =["London Paris London","Paris Paris London","London Paris London"]
cv = CountVectorizer()
count_matrix = cv.fit_transform(text)
print(count_matrix.toarray())
sim = cosine_similarity(count_matrix)
"""

#print(df[df.title == "Spectre"]["index"].values[0])

features = ['keywords','cast','genres','director']
for feature in features:
    df[feature] = df[feature].fillna('')
def combine(row):
        return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director'] 
df['combined'] = df.apply(combine,axis=1)
cv = CountVectorizer()

count_matrix = cv.fit_transform(df['combined'])
 
sim = cosine_similarity(count_matrix)
#print(sim)
user = "Avatar"

movie_index = get_index_from_title(user)

sim_movies = list(enumerate(sim[movie_index]))

sorted_movie = sorted(sim_movies,key = lambda x:x[1],reverse = True)

i = 0
for mov in sorted_movie:
    print(get_title_from_index(mov[0]))
    if(i == 5):
        break
    i+=1
