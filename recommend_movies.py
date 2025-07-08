
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import csr_matrix
import pickle

movies = pd.read_csv(r"dataset\movies.csv")
ratings = pd.read_csv(r"dataset\ratings.csv")

df = pd.merge(ratings, movies, on="movieId")
user_movie_matrix = df.pivot_table(index='userId', columns='movieId', values='rating')
user_movie_matrix.fillna(0, inplace=True)
X = user_movie_matrix.values


def recommend_movies_user_based(user_vector, k=3, top_n=5):
    

    user_vector = user_vector.reshape(1, -1)    

    # Load the model
    try:
        with open('knn.pkl', 'rb') as file:
            model = pickle.load(file)
    except FileNotFoundError:
        print("Model file not found. Please ensure the model is saved correctly.")
        return []

    distances, indices = model.kneighbors(user_vector, n_neighbors=k+1)
    similar_user_indices = indices.flatten()[1:]
    similar_user_ids = user_movie_matrix.index[similar_user_indices]
    print(f"Similar User IDs: {similar_user_ids}")
    user_seen_movies = set(index for index , rating in enumerate(user_vector[0]) if rating > 0)
    movie_scores = {}

    for sim_user in similar_user_ids:
        sim_user_ratings = user_movie_matrix.loc[sim_user]
        for movie, rating in sim_user_ratings.items():
            if movie not in user_seen_movies and rating > 0:
                if movie not in movie_scores:
                    movie_scores[movie] = [rating]
                else:
                    movie_scores[movie].append(rating)

    avg_scores = {movie: np.mean(ratings) for movie, ratings in movie_scores.items()}
    recommended_movies = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    for title, score in recommended_movies:
        print(f"{title} (Avg Rating: {score:.2f})")

    recommended_movies = [(movies[movies["movieId"] == movie]["title"].values[0]) for movie, _ in recommended_movies]
    return recommended_movies
