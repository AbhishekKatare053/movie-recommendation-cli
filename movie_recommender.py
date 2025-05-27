import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack
import joblib
import os

# Extract cast names
def extract_cast_names(cast_str):
    try:
        cast_list = ast.literal_eval(cast_str)
        return ' '.join([actor['name'] for actor in cast_list[:10]])
    except:
        return ''

# Parse genres
def parse_genres(df):
    def get_genre_names(genres_str):
        try:
            genres_list = ast.literal_eval(genres_str)
            return ' '.join(genre['name'] for genre in genres_list)
        except:
            return ''
    df['genres_parsed'] = df['genres'].apply(get_genre_names)
    return df

# Load movie metadata
def load_movies_metadata(path='data/movies_metadata.csv'):
    try:
        df = pd.read_csv(path, low_memory=False)
        df = df[['id', 'title', 'genres', 'overview', 'original_language']].dropna()
        df = df[df['overview'] != '']
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        df.dropna(subset=['id'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"[INFO] Cleaned metadata: {df.shape}")
        return df
    except FileNotFoundError:
        print("[ERROR] movies_metadata.csv not found.")
        return None

# Load credits data
def load_credits_data(path='data/credits.csv'):
    try:
        df = pd.read_csv(path)
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        df['cast_names'] = df['cast'].apply(extract_cast_names)
        print(f"[INFO] Credits data: {df.shape}")
        return df
    except FileNotFoundError:
        print("[ERROR] credits.csv not found.")
        return None

# Merge metadata and credits
def merge_data(metadata_df, credits_df):
    df = pd.merge(metadata_df, credits_df[['id', 'cast_names']], on='id', how='left')
    print(f"[INFO] Merged dataset: {df.shape}")
    df.to_csv("data/final_merged.csv", index=False)
    print("[INFO] Merged CSV saved at 'data/final_merged.csv'")
    return df

# Vectorization
def vectorize_genres(df):
    cv = CountVectorizer()
    return cv.fit_transform(df['genres_parsed'])

def vectorize_overviews(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    return tfidf.fit_transform(df['overview'])

# Similarity model
def build_similarity_model(genre_matrix, tfidf_matrix):
    combined = hstack([genre_matrix, tfidf_matrix])
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(combined)
    return model, combined

# Recommendations
def get_movie_recommendations(title, knn_model, combined_features, df, n=5):
    try:
        idx = df[df['title'].str.lower() == title.lower()].index[0]
    except IndexError:
        print(f"[ERROR] '{title}' not found.")
        return []
    _, indices = knn_model.kneighbors(combined_features[idx], n_neighbors=n+1)
    recs = [df.iloc[i]['title'] for i in indices.flatten()[1:]]
    print(f"\n🎥 Top {n} recommendations for '{title}':")
    for i, name in enumerate(recs, 1):
        print(f"{i}. {name}")
    return recs

# ====== MAIN EXECUTION ======
if __name__ == "__main__":
    print("🔁 Initializing Movie Recommender System...")

    metadata_df = load_movies_metadata()
    credits_df = load_credits_data()
    if metadata_df is None or credits_df is None:
        exit()

    metadata_df = parse_genres(metadata_df)
    df = merge_data(metadata_df, credits_df)

    if os.path.exists('models/knn_model.joblib'):
        print("[INFO] Loading saved model...")
        knn_model = joblib.load('models/knn_model.joblib')
        combined_features = joblib.load('models/combined_features.joblib')
    else:
        print("[INFO] No saved model found. Building from scratch...")
        genre_matrix = vectorize_genres(df)
        tfidf_matrix = vectorize_overviews(df)
        knn_model, combined_features = build_similarity_model(genre_matrix, tfidf_matrix)

        os.makedirs('models', exist_ok=True)
        joblib.dump(knn_model, 'models/knn_model.joblib')
        joblib.dump(combined_features, 'models/combined_features.joblib')
        joblib.dump(genre_matrix, 'models/genre_matrix.joblib')
        joblib.dump(tfidf_matrix, 'models/tfidf_matrix.joblib')
        print("[INFO] Model and features saved to /models/")

    # ========== INTERFACE ==========
    print("\n🎮 Welcome to Movie Recommender System 🎮")
    while True:
        print("\nChoose search mode:")
        print("1. Search by Movie Title")
        print("2. Search by Genre")
        print("3. Search Bollywood Movies")
        print("4. Search by Mood")
        print("5. Search by Actor/Actress")
        print("6. Exit")
        choice = input("Enter your choice (1-6): ").strip()

        if choice == '1':
            title = input("\n🎬 Enter movie title: ").strip()
            get_movie_recommendations(title, knn_model, combined_features, df)

        elif choice == '2':
            genre_input = input("\n🎭 Enter genre: ").strip().lower()
            lang = input("🌍 Bollywood or Hollywood? ").strip().lower()
            lang_filter = 'hi' if 'bollywood' in lang else 'en'
            limit = input("📊 How many movies? (10/15/all): ").strip()
            limit = None if limit == 'all' else int(limit) if limit.isdigit() else 10

            filtered = df[(df['genres_parsed'].str.lower().str.contains(genre_input)) &
                          (df['original_language'] == lang_filter)]
            if filtered.empty:
                print("[ERROR] No matches found.")
                continue
            print(f"\n🎭 {genre_input.title()} Movies:")
            for i, title in enumerate(filtered['title'].head(limit), 1):
                print(f"{i}. {title}")

        elif choice == '3':
            limit = input("\n📊 Number of Bollywood movies? (10/15/all): ").strip()
            limit = None if limit == 'all' else int(limit) if limit.isdigit() else 10
            bollywood = df[df['original_language'] == 'hi']
            print("\n🎬 Top Bollywood Movies:")
            for i, title in enumerate(bollywood['title'].head(limit), 1):
                print(f"{i}. {title}")

        elif choice == '4':
            mood = input("\n😊 Enter mood (e.g. happy/sad): ").strip().lower()
            lang = input("🌍 Bollywood or Hollywood? ").strip().lower()
            lang_filter = 'hi' if 'bollywood' in lang else 'en'
            limit = input("📊 Number of movies? (10/15/all): ").strip()
            limit = None if limit == 'all' else int(limit) if limit.isdigit() else 10
            mood_df = df[(df['overview'].str.lower().str.contains(mood)) &
                         (df['original_language'] == lang_filter)]
            print(f"\n🎭 Mood: {mood.title()} Movies:")
            for i, title in enumerate(mood_df['title'].head(limit), 1):
                print(f"{i}. {title}")

        elif choice == '5':
            actor = input("\n🎬 Enter actor/actress name: ").strip().lower()
            lang = input("🌍 Bollywood or Hollywood? ").strip().lower()
            lang_filter = 'hi' if 'bollywood' in lang else 'en'
            limit = input("📊 Number of movies? (10/15/all): ").strip()
            limit = None if limit == 'all' else int(limit) if limit.isdigit() else 10
            actor_df = df[(df['cast_names'].str.lower().str.contains(actor, na=False)) &
                          (df['original_language'] == lang_filter)]
            if actor_df.empty:
                print("[ERROR] No movies found.")
                continue
            print(f"\n🎬 Movies with '{actor.title()}':")
            for i, title in enumerate(actor_df['title'].head(limit), 1):
                print(f"{i}. {title}")

        elif choice == '6':
            print("👋 Exiting. Thank you for using the system.")
            break

        else:
            print("[ERROR] Invalid choice. Please try again.")
