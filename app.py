import streamlit as st
import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@st.cache_data
def load_data():
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    return ratings

@st.cache_resource
def train_model(data):
    reader = Reader()
    dataset = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(dataset, test_size=0.2)
    algo = SVD()
    algo.fit(trainset)
    return algo

@st.cache_data
def load_movies():
    movies = pd.read_csv('ml-25m/movies.csv')
    movie_titles = dict(zip(movies['movieId'], movies['title']))
    return movie_titles

@st.cache_data
def get_movie_id(movie_titles, selected_movie):
    movie_ids = {v: k for k, v in movie_titles.items()}
    return movie_ids[selected_movie]

@st.cache_data
def compute_similarities(_algo, movie_titles):
    # Filter out movies not in the trainset
    movie_ids_in_trainset = {movie_id for movie_id in movie_titles if movie_id in algo.trainset._raw2inner_id_items}
    # Initialize a matrix to hold the latent factors
    latent_factors = np.zeros(algo.qi.shape)

    for movie_id in movie_ids_in_trainset:
        inner_id = algo.trainset.to_inner_iid(movie_id)
        latent_factors[inner_id] = algo.qi[inner_id]

    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(latent_factors)
    return similarity_matrix, movie_ids_in_trainset


def recommend_movies(algo, selected_movie_id, movie_titles, similarity_matrix, movie_ids_in_trainset, num_recommendations=10):
    # Ensure the selected movie is in the trainset
    if selected_movie_id not in movie_ids_in_trainset:
        return ["Selected movie is not in the training set."]

    movie_inner_id = algo.trainset.to_inner_iid(selected_movie_id)
    similarity_scores = list(enumerate(similarity_matrix[movie_inner_id]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    movie_neighbors = [algo.trainset.to_raw_iid(inner_id) for inner_id, _ in similarity_scores[1:num_recommendations+1]]
    recommended_movies = [movie_titles[movie_id] for movie_id in movie_neighbors]
    return recommended_movies

data = load_data()
algo = train_model(data)
movie_titles = load_movies()
similarity_matrix, movie_ids_in_trainset = compute_similarities(algo, movie_titles)


st.title("Movie Recommendation System")

selected_movie = st.selectbox("Select a movie:", list(movie_titles.values()))

selected_movie_id = get_movie_id(movie_titles, selected_movie)

if st.button("Recommend"):
    recommendations = recommend_movies(algo, selected_movie_id, movie_titles, similarity_matrix, movie_ids_in_trainset)
    st.write("Movies recommended based on your selection:")
    for i, movie in enumerate(recommendations):
        st.write(f"{i+1}. {movie}")
