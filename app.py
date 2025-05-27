import streamlit as st
import pandas as pd
import joblib

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv("data/final_merged.csv")
    return df

@st.cache_resource
def load_model():
    knn_model = joblib.load('models/knn_model.joblib')
    combined_features = joblib.load('models/combined_features.joblib')
    return knn_model, combined_features

df = load_data()
knn_model, combined_features = load_model()

st.title("🎬 Movie Recommendation System")

movie_title = st.text_input("Enter Movie Title:")

if st.button("Get Recommendations"):
    if movie_title.strip() == "":
        st.warning("Please enter a movie title!")
    else:
        try:
            idx = df[df['title'].str.lower() == movie_title.lower()].index[0]
            distances, indices = knn_model.kneighbors(combined_features[idx], n_neighbors=6)
            recommendations = df.iloc[indices.flatten()[1:]]['title'].values
            st.success(f"Top recommendations for '{movie_title}':")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        except IndexError:
            st.error(f"Movie titled '{movie_title}' not found in database.")
