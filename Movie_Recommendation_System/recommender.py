from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- File Paths ----------------
BASE_DIR = Path(__file__).resolve().parent

movies = pd.read_csv(BASE_DIR / "dataset" / "movies.csv")
ratings = pd.read_csv(BASE_DIR / "dataset" / "ratings.csv")

# ---------------- Create User-Movie Matrix ----------------
movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

# ---------------- Calculate Similarity ----------------
similarity = cosine_similarity(movie_matrix.T)

similarity_df = pd.DataFrame(
    similarity,
    index=movie_matrix.columns,
    columns=movie_matrix.columns
)


# ---------------- Recommendation Function ----------------
def recommend(movie_name):

    movie = movies[movies["title"] == movie_name]

    if movie.empty:
        return pd.DataFrame()

    movie_id = movie.iloc[0]["movieId"]

    scores = similarity_df[movie_id].sort_values(ascending=False)

    recommendations = []

    for mid in scores.index[1:7]:

        movie_info = movies[movies["movieId"] == mid]

        recommendations.append({
            "Movie": movie_info.iloc[0]["title"],
            "Genre": movie_info.iloc[0]["genre"],
            "Year": movie_info.iloc[0]["year"],
            "Similarity": round(scores[mid], 2)
        })

    return pd.DataFrame(recommendations)