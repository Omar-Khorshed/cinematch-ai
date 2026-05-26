import pandas as pd
import ast
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi
import streamlit as st
import os

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_crew(text):
    names = []
    try:
        crew_list = ast.literal_eval(text)
        for item in crew_list:
            if item['job'] == 'Director':
                names.append(item['name'])
    except:
        pass
    return " ".join(names)

def clean_languages(text):
    names = []
    try:
        lang_list = ast.literal_eval(text)
        for item in lang_list:
            names.append(item['name'])
    except:
        pass
    return " ".join(names)

def clean_genres(text):
    names = []
    try:
        genres_list = ast.literal_eval(text)
        for item in genres_list:
            names.append(item['name'])
    except:
        pass
    return " ".join(names)

def clean_cast(text):
    names = []
    try:
        cast_list = ast.literal_eval(text)
        for item in cast_list[:5]:
            names.append(item['name'])
    except:
        pass
    return " ".join(names)

def clean_keywords(text):
    names = []
    try:
        keywords_list = ast.literal_eval(text)
        for item in keywords_list:
            names.append(item['name'])
    except:
        pass
    return " ".join(names)

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

def preprocess_title(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    return text

@st.cache_resource(show_spinner="Loading models and processing data...")
def load_and_prepare_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.join(base_dir, 'tmdb_5000_movies.csv')
    credits_path = os.path.join(base_dir, 'tmdb_5000_credits.csv')
    
    try:
        movies = pd.read_csv(movies_path)
    except UnicodeDecodeError:
        movies = pd.read_csv(movies_path, encoding='latin-1')
        
    try:
        credits = pd.read_csv(credits_path)
    except UnicodeDecodeError:
        credits = pd.read_csv(credits_path, encoding='latin-1')
    
    movies = movies.merge(credits, left_on='id', right_on='movie_id')
    movies.drop(columns=['movie_id'], inplace=True)
    movies.drop(columns=['title_y'], inplace=True)
    movies.rename(columns={'title_x': 'title'}, inplace=True)
    
    movies['director'] = movies['crew'].apply(clean_crew)
    movies['spoken_languages_clean'] = movies['spoken_languages'].apply(clean_languages)
    movies['genres_clean'] = movies['genres'].apply(clean_genres)
    movies['main_cast'] = movies['cast'].apply(clean_cast)
    movies['keywords_clean'] = movies['keywords'].apply(clean_keywords)
    
    movies['clean_title'] = movies['title'].apply(preprocess_title)
    movies['clean_overview'] = movies['overview'].apply(preprocess)
    movies['clean_cast'] = movies['main_cast'].apply(preprocess_title)
    movies['clean_director'] = movies['director'].apply(preprocess_title)
    
    # Fill NA values
    movies['overview'] = movies['overview'].fillna('')
    movies['genres_clean'] = movies['genres_clean'].fillna('Unknown')
    movies['director'] = movies['director'].fillna('Unknown')
    movies['main_cast'] = movies['main_cast'].fillna('Unknown')
    movies['vote_average'] = movies['vote_average'].fillna(0.0)
    movies['popularity'] = movies['popularity'].fillna(0.0)
    
    # Independent BM25 Setup for Weights Support
    tokenized_title = [doc.split() for doc in movies['clean_title']]
    tokenized_overview = [doc.split() for doc in movies['clean_overview']]
    tokenized_cast = [doc.split() for doc in movies['clean_cast']]
    tokenized_director = [doc.split() for doc in movies['clean_director']]
    
    bm25_title = BM25Okapi(tokenized_title)
    bm25_overview = BM25Okapi(tokenized_overview)
    bm25_cast = BM25Okapi(tokenized_cast)
    bm25_director = BM25Okapi(tokenized_director)
    
    models = {
        'bm25_title': bm25_title,
        'bm25_overview': bm25_overview,
        'bm25_cast': bm25_cast,
        'bm25_director': bm25_director
    }
    
    return movies, models

def bm25_search_func(query, movies, models, title_w=2.0, overview_w=1.0, cast_w=1.5, director_w=1.5, top_k=20):
    clean_query_title = preprocess_title(query)
    clean_query = preprocess(query)
    
    tokenized_query_title = clean_query_title.split()
    tokenized_query = clean_query.split()
    
    # Get scores for each field
    score_title = models['bm25_title'].get_scores(tokenized_query_title)
    score_overview = models['bm25_overview'].get_scores(tokenized_query)
    score_cast = models['bm25_cast'].get_scores(tokenized_query_title)
    score_director = models['bm25_director'].get_scores(tokenized_query_title)
    
    # Apply weights
    final_scores = (
        title_w * score_title +
        overview_w * score_overview +
        cast_w * score_cast +
        director_w * score_director
    )
    
    top_indices = final_scores.argsort()[-top_k:][::-1]
    
    results = movies.iloc[top_indices][[
        'title', 'genres_clean', 'main_cast', 'director', 'vote_average', 'popularity', 'overview'
    ]].copy()
    results['score'] = final_scores[top_indices]
    
    # Return results that have a score > 0
    results = results[results['score'] > 0]
    return results
