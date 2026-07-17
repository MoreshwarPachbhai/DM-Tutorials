from pathlib import Path
import streamlit as st

from recommender import movies, ratings, recommend

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CSS ----------------
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ---------------- Title ----------------
st.title("🎬 Movie Recommendation System")

st.write(
    "Recommend similar movies using **Collaborative Filtering (Cosine Similarity)**"
)

st.divider()

# ---------------- Metrics ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🎬 Movies", len(movies))

with col2:
    st.metric("👤 Users", ratings["userId"].nunique())

with col3:
    st.metric("⭐ Ratings", len(ratings))

st.divider()

# ---------------- Movie Selection ----------------
movie = st.selectbox(
    "Select a Movie",
    sorted(movies["title"].unique())
)

if st.button("🎯 Recommend Movies", use_container_width=True):

    result = recommend(movie)

    if result.empty:
        st.error("Movie not found.")
    else:
        st.success("Top 6 Similar Movies")

        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )

st.divider()

# ---------------- Dataset ----------------
with st.expander("📂 View Movie Dataset"):

    st.dataframe(
        movies,
        use_container_width=True,
        hide_index=True
    )