#!/usr/bin/env python
# coding: utf-8

# **Netflix Movie Recommendation**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('netflix_titles.csv')


# In[ ]:


data.head()


# In[ ]:


movie_data = data.loc[data.type=='Movie',:].reset_index()


# In[ ]:


movie_data.title = movie_data.title.str.lower()
movie_data['index'] = movie_data.index
movie_data.head()


# In[ ]:


movie_data.columns


# In[ ]:


features = ['director', 'cast', 'country', 'description', 'listed_in']


# In[ ]:


def combine_features(row):
    return row['director'] +" "+row['cast']+" "+row["country"]+" "+row["description"]+" "+row["listed_in"]


# In[ ]:


for feature in features:
    movie_data[feature] = movie_data[feature].fillna('')
    
movie_data["combined_features"] = movie_data.apply(combine_features,axis=1)
cv = CountVectorizer()
count_matrix = cv.fit_transform(movie_data["combined_features"])


# In[ ]:


cosine_sim = cosine_similarity(count_matrix)


# In[ ]:


def get_title_from_index(index):
    return movie_data[movie_data.index == index]["title"].values[0]


# In[ ]:


def get_title_from_index(df, index):
    return df[df.index == index]["title"].values[0]


# In[ ]:


def get_index_from_title(df, title):
    return df[df.title == title]["index"].values[0]


# In[ ]:


def recommend(movie_user_likes):
    try:
        movie_user_likes = movie_user_likes.lower()
        movie_index = get_index_from_title(movie_data, movie_user_likes)
        similar_movies =  list(enumerate(cosine_sim[movie_index]))
        sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
        i=0
        print("Top 5 similar movies to "+movie_user_likes+" are:\n")
        recommended_movies = []
        for element in sorted_similar_movies:
            recommended_movies.append(get_title_from_index(movie_data, element[0]))
            i=i+1
            if i>=5:
                break
        return recommended_movies
    except:
        return('Movie not found on Netflix. Please retry!')


# In[ ]:



# In[ ]:




