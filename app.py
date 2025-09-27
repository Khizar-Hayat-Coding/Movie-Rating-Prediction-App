import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.markdown("""
    <style>
    .purple-heading {
        color: #800080;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5em;
    }
    .stButton>button {
        background-color: #800080 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold;
        border: none;
    }
    .pred-box {
        background-color: #f3e6ff;
        padding: 10px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #800080;
        margin-top: 1em;
        margin-bottom: 1em;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('movie_rating_model.pkl', 'rb') as f:
        bundle = pickle.load(f)
    return bundle

bundle = load_model()
model = bundle['model']
genre_means = bundle['genre_means']
director_means = bundle['director_means']
actor_means = bundle['actor_means']
genres = bundle['genres']
directors = bundle['directors']
actors = bundle['actors']

st.markdown('<div class="purple-heading">🎬 Movie Rating Prediction App</div>', unsafe_allow_html=True)
st.write("Enter movie details to predict its IMDb rating.")

col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Release Year", min_value=1950, max_value=2050, value=2022)
    duration = st.number_input("Duration (minutes)", min_value=30, max_value=300, value=120)
    votes = st.number_input("Number of Votes", min_value=1, max_value=10_000_000, value=15000)

with col2:
    genre = st.selectbox("Genre", sorted(genres))
    director = st.selectbox("Director", sorted(directors))
    actor = st.selectbox("Lead Actor", sorted(actors))

genre_mean_rating = genre_means.get(genre, np.mean(list(genre_means.values())))
director_encoded = director_means.get(director, np.mean(list(director_means.values())))
actor_encoded = actor_means.get(actor, np.mean(list(actor_means.values())))

input_df = pd.DataFrame([{
    'Year': year,
    'Votes': votes,
    'Duration': duration,
    'Genre_mean_rating': genre_mean_rating,
    'Director_encoded': director_encoded,
    'Actor_encoded': actor_encoded
}])

predict_btn = st.button("Predict Rating")
if predict_btn:
    pred = model.predict(input_df)[0]
    st.markdown(
        f'<div class="pred-box">Predicted Rating: {pred:.2f}</div>',
        unsafe_allow_html=True
    )
 