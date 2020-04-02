from flask import Flask, render_template, request
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('netflix_titles.csv')

movie_data = data.loc[data.type=='Movie',:].reset_index()

movie_data.title = movie_data.title.str.lower()
movie_data['index'] = movie_data.index
features = ['director', 'cast', 'country', 'description', 'listed_in']

def combine_features(row):
    return row['director'] +" "+row['cast']+" "+row["country"]+" "+row["description"]+" "+row["listed_in"]

for feature in features:
    movie_data[feature] = movie_data[feature].fillna('')
    
movie_data["combined_features"] = movie_data.apply(combine_features,axis=1)
cv = CountVectorizer()
count_matrix = cv.fit_transform(movie_data["combined_features"])

cosine_sim = cosine_similarity(count_matrix)

def get_title_from_index(index):
    return movie_data[movie_data.index == index]["title"].values[0]


def get_title_from_index(df, index):
    return df[df.index == index]["title"].values[0]


def get_index_from_title(df, title):
    return df[df.title == title]["index"].values[0]

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
            recommended_movies.append(get_title_from_index(movie_data, element[0]).title())
            i=i+1
            if i>=5:
                break
        return recommended_movies
    except:
        return('Movie not found on Netflix. Please retry!')


app = Flask(__name__)


@app.route('/home')
def home():
    return render_template('home.html')

#app.route("/recommend")

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		input_movie = request.form.get('movie')
	movies = recommend(input_movie)

	return render_template('output.html',movies=movies, input_movie = input_movie.title())


if __name__ == '__main__':
    app.run()
