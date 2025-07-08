from fastapi import FastAPI, Query
from typing import List
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from scipy.sparse import csr_matrix
from recommend_movies import recommend_movies_user_based
import requests
app = FastAPI()
app.add_middleware(
CORSMiddleware,
allow_origins=["*"],  # List of allowed origins
allow_credentials=True,  # Allow cookies and credentials
allow_methods=["*"],     # Allow all HTTP methods
allow_headers=["*"],     # Allow all headers
)


# Pre-load CSV once
df = pd.read_csv("dataset/movies.csv")
rating = pd.read_csv("dataset/ratings.csv")

# http://localhost:8080/getMovies?moviesCount=5
@app.get("/getMovies", response_model=dict)
async def get_movies(moviesCount: int = Query(5, ge=1, le=100)):
    """
    Returns a random list of movie titles with indices.

    - moviesCount: number of movies to return (default 5, min 1, max 100)
    """

    filtered_df = df[df["movieId"].isin(rating["movieId"])]
    sample_df = filtered_df.sample(n=moviesCount)
    movie_names = sample_df["title"].to_list()
    movie_ids = sample_df["movieId"].to_list()
    movie_indices = [df.loc[df['movieId'] == id].index[0] for id in movie_ids]

    # size =df.shape[0]  # Total number of movies in the dataset 
    size = 5401
    # Make a list of dicts
    movies_list = [
    {
        "index": int(idx),
        "id": int(mid),
        "title": name
    }
    for idx, mid, name in zip(movie_indices, movie_ids, movie_names)
]

    response = {
        "length": size,
        "movies": movies_list
    }
    
    return response

class Movie(BaseModel):
    index : int
    id : int
    title : str
    rating : int = 0

class MovieMetaData(BaseModel):
    length: int
    movies : List[Movie]

@app.post("/recommendMovies")
async def recommend_movies( movies : MovieMetaData):
    user_vector = np.zeros((1,movies.length))
    for movie in movies.movies:
        user_vector[0,movie.index] = movie.rating
    recommended_movies = recommend_movies_user_based(user_vector, k=5, top_n=20)
    import urllib.parse
    OMDB_KEY = 'fdd792f1'
    response = []
    # Example: use the movie title for search; adjust as needed
    for movie in recommended_movies:
        movie = urllib.parse.quote(movie[:-7])
        url = f"https://www.omdbapi.com/?apikey={OMDB_KEY}&t={movie}"
        res = requests.get(url)
        if res.status_code == 200:
            res = res.json()
            if res.get("Response") == "True":   
                response.append(res)
    return response






