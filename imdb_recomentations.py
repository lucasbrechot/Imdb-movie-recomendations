# Import Pandas
import pandas as pd
import json
import numpy as np
from ast import literal_eval

movies = pd.read_csv('/Users/lucasbrechot/Imdb/archive/movies_metadata.csv')
ratings = pd.read_csv('/Users/lucasbrechot/Imdb/archive/ratings.csv')
keywords = pd.read_csv('/Users/lucasbrechot/Imdb/archive/keywords.csv')
credits = pd.read_csv('/Users/lucasbrechot/Imdb/archive/credits.csv')

movies = movies.drop([19730, 29503, 35587])

movies["id"] = pd.to_numeric(movies["id"])
keywords["id"] = pd.to_numeric(keywords["id"])

movies = pd.merge(left=movies,right=keywords,how='inner',left_on='id',right_on='id')
movies = pd.merge(left=movies,right=credits[["id","cast"]],how='inner',left_on='id',right_on='id')


# Parse the stringified features into their corresponding python objects

features = ['cast','keywords', 'genres']
for feature in features:
    movies[feature] = movies[feature].apply(literal_eval)

def get_director(x):
    
    for i in x:
        if i["job"] == 'Director':
            return i['name']
        
        return np.nan

def get_list(x):
    if isinstance(x,list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

movies["director"] = movies["crew"].apply(get_director)

features = ['cast','keywords', 'genres','crew']
for feature in features:
    movies[feature] = movies[feature].apply(get_list)


def clean_data(x):
    if isinstance(x,list):
        return [str.lower(i.replace(" ","")) for i in x]
    else:
        
        if isinstance(x,str):
            return str.lower(x.replace(" ",""))
        else:
            
            return ''

features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    movies[feature] = movies[feature].apply(clean_data)


def create_soup(x):    
    return ' '.join(x["genres"]) + ' ' + ' '.join(x["cast"])+ ' ' + ' '.join(x["keywords"])+ ' ' + x["director"]


movies["soup"] = movies.apply(create_soup, axis=1)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(movies['soup'])


cosine_sim2 = cosine_similarity(count_matrix,count_matrix)


# Reset index of your main DataFrame and construct reverse mapping as before
movies = movies.reset_index()
indices = pd.Series(movies.index, index=movies['title'])

def get_recommendations(title, cosine_sim=cosine_sim2):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies['title'].iloc[movie_indices]







# Recomendations based on grade

import pandas as pd
import json

movies = pd.read_csv('/Users/lucasbrechot/Imdb/archive/movies_metadata.csv')
ratings = pd.read_csv('/Users/lucasbrechot/Imdb/archive/ratings.csv')
keywords = pd.read_csv('/Users/lucasbrechot/Imdb/archive/keywords.csv')
credits = pd.read_csv('/Users/lucasbrechot/Imdb/archive/credits.csv')

## Simple recomendations system

C = movies["vote_average"].mean()
m = movies["vote_count"].quantile(0.90)

q_movies = movies[movies["vote_count"] >= m]

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

q_movies = q_movies.sort_values(by='score', ascending=False)

q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20)
