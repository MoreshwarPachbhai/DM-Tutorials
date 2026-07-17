import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

similarity = cosine_similarity(movie_matrix.T)

similarity_df = pd.DataFrame(
    similarity,
    index=movie_matrix.columns,
    columns=movie_matrix.columns
)


def recommend(movie_name):

    movie = movies[movies["title"] == movie_name]

    if movie.empty:
        return pd.DataFrame()

    movie_id = movie.iloc[0]["movieId"]

    scores = similarity_df[movie_id].sort_values(ascending=False)

    recommendations = []

    for mid in scores.index[1:7]:

        title = movies[movies["movieId"] == mid]["title"].values[0]

        genre = movies[movies["movieId"] == mid]["genre"].values[0]

        recommendations.append({
            "Movie": title,
            "Genre": genre,
            "Similarity": round(scores[mid],2)
        })

    return pd.DataFrame(recommendations)