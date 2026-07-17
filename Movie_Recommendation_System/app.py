import streamlit as st
import pandas as pd
from recommender import recommend
from recommender import movies

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

st.title("🎬 Movie Recommendation System")

st.write("Find similar movies using Collaborative Filtering")

col1,col2,col3=st.columns(3)

with col1:
    st.metric("Movies",len(movies))

with col2:
    st.metric("Users",7)

with col3:
    st.metric("Ratings",35)

st.divider()

movie=st.selectbox(
    "Select Movie",
    movies["title"]
)

if st.button("Recommend Movies"):

    result=recommend(movie)

    st.success("Top Recommended Movies")

    st.dataframe(
        result,
        use_container_width=True
    )

st.divider()

st.subheader("Movie Dataset")

st.dataframe(
    movies,
    use_container_width=True
)