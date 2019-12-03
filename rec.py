# -*- coding: utf-8 -*-
"""
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

#print(df[df.title == "Spectre"]["index"].values[0])

features = ['keywords','cast','genres','director']
for feature in features:
    df[feature] = df[feature].fillna('')
def combine(row):
        return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director'] 

#Apply to entire dataset and create new column -> combined
df['combined'] = df.apply(combine,axis=1)
cv = CountVectorizer()


count_matrix = cv.fit_transform(df['combined'])
 
sim = cosine_similarity(count_matrix)
#print(sim)
user = input("Enter movie: ")

movie_index = get_index_from_title(user)

sim_movies = list(enumerate(sim[movie_index]))

sorted_movie = sorted(sim_movies,key = lambda x:x[1],reverse = True)


#Print movies sorted by cosine score for a row taht matches user input
i = 0
print("Recommended movies are :-")
for mov in sorted_movie:
    print(get_title_from_index(mov[0]))
    if(i == 5):
        break
    i+=1
