import streamlit as st
import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# -------------------- Load Data --------------------
@st.cache_data
def load_ratings(path="ml-latest-small/ratings.csv"):
    return pd.read_csv(path)

@st.cache_data
def load_movies(path="ml-latest-small/movies.csv"):
    movies = pd.read_csv(path)
    movies = movies.reset_index(drop=True)
    return movies

# -------------------- Train Model --------------------
@st.cache_resource
def train_svd_model(ratings_df):
    reader = Reader(rating_scale=(ratings_df.rating.min(), ratings_df.rating.max()))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.2)
    algo = SVD()
    algo.fit(trainset)
    return algo

@st.cache_data
def compute_cf_similarities(_algo, movie_id_list):
    trainset = _algo.trainset
    inner_to_raw = [None] * trainset.n_items
    raw_to_inner = {}
    for raw, inner in trainset._raw2inner_id_items.items():
        inner_to_raw[inner] = raw
        raw_to_inner[raw] = inner
    latent = _algo.qi.copy()
    similarity_matrix = cosine_similarity(latent)
    return similarity_matrix, inner_to_raw, raw_to_inner

@st.cache_data
def compute_content_similarity(movies_df):
    movies = movies_df.copy()
    movies['genres'] = movies['genres'].fillna('')
    cv = CountVectorizer(tokenizer=lambda x: x.split('|'))
    genre_matrix = cv.fit_transform(movies['genres'])
    sim = cosine_similarity(genre_matrix, genre_matrix)
    return sim

@st.cache_data
def compute_popularity_stats(ratings_df):
    grp = ratings_df.groupby('movieId').agg(count=('rating','size'), avg_rating=('rating','mean')).reset_index()
    return grp

def normalize_arr(arr):
    arr = np.array(arr, dtype=float)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

def hybrid_recommend(selected_movie_title,
                     movies_df,
                     movie_title_to_id,
                     cf_sim_matrix,
                     cf_inner_to_raw,
                     cf_raw_to_inner,
                     content_sim_matrix,
                     pop_stats,
                     alpha=0.5,
                     boost_popularity=True,
                     n_recs=10):

    try:
        selected_raw_id = movie_title_to_id[selected_movie_title]
    except KeyError:
        return ["Movie not found."]

    # collaborative part
    if selected_raw_id in cf_raw_to_inner:
        sel_inner = cf_raw_to_inner[selected_raw_id]
        collab_scores = cf_sim_matrix[sel_inner]
        collab_raw_score = {cf_inner_to_raw[i]: float(collab_scores[i]) for i in range(len(collab_scores))}
        collab_aligned = movies_df['movieId'].map(lambda mid: collab_raw_score.get(mid, 0.0)).fillna(0.0).values
    else:
        collab_aligned = np.zeros(len(movies_df), dtype=float)
    collab_norm = normalize_arr(collab_aligned)

    # content part
    sel_idx = movies_df.index[movies_df['movieId'] == selected_raw_id]
    if len(sel_idx) == 0:
        content_aligned = np.zeros(len(movies_df), dtype=float)
    else:
        sel_idx = sel_idx[0]
        content_scores = content_sim_matrix[sel_idx]
        content_aligned = np.array(content_scores, dtype=float)
    content_norm = normalize_arr(content_aligned)

    # hybrid score
    hybrid_score = alpha * collab_norm + (1 - alpha) * content_norm

    # popularity boost
    if boost_popularity:
        pop_map = pop_stats.set_index('movieId')['count'].to_dict()
        avg_map = pop_stats.set_index('movieId')['avg_rating'].to_dict()
        pop_counts = movies_df['movieId'].map(lambda m: pop_map.get(m, 0)).fillna(0).values
        avg_ratings = movies_df['movieId'].map(lambda m: avg_map.get(m, np.nan)).fillna(np.nan).values
        pop_norm = normalize_arr(pop_counts)
        rating_norm = normalize_arr(np.nan_to_num(avg_ratings, nan=np.nanmean(avg_ratings)))
        pop_modifier = 0.6 * pop_norm + 0.4 * rating_norm
        final_score = 0.8 * hybrid_score + 0.2 * pop_modifier
    else:
        final_score = hybrid_score

    selected_mask = (movies_df['movieId'] == selected_raw_id)
    final_score[selected_mask.values] = -1

    top_idx = np.argsort(final_score)[::-1][:n_recs]
    recommended_titles = movies_df.iloc[top_idx]['title'].tolist()
    return recommended_titles

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Movie Matchmaker", page_icon="🎬")

st.title("🍿 Your Personal Movie Matchmaker")

st.markdown("""
Welcome! 👋  
Not sure what to watch next?  
Just pick a movie you like, and we’ll suggest others you’ll probably enjoy.  

Our system looks at **two things**:
- Movies that **people like you** enjoyed  
- Movies that are **similar in theme/genre**  

We then mix them together to get the best of both worlds ✨
""")

ratings = load_ratings()
movies = load_movies()

with st.spinner("Getting ready..."):
    algo = train_svd_model(ratings)

cf_sim_matrix, cf_inner_to_raw, cf_raw_to_inner = compute_cf_similarities(algo, movies['movieId'].tolist())
content_sim_matrix = compute_content_similarity(movies)
pop_stats = compute_popularity_stats(ratings)

movie_title_to_id = dict(zip(movies['title'], movies['movieId']))

st.subheader("🎥 Step 1: Pick a movie you like")
selected_movie = st.selectbox("", movies['title'].tolist())

st.subheader("⚖️ Step 2: How should we match your movie?")
st.markdown("""
- **More like people’s choices** → Stronger recommendations based on what other viewers liked.  
- **More like the movie itself** → Stronger match on theme, style, or genre.  
""")
alpha = st.slider("Balance:", 0.0, 1.0, 0.6, 0.05, help="0 = by movie similarity, 1 = by viewer choices")

boost_pop = st.checkbox("Give more weight to popular & well-rated movies", value=True)

n_recs = st.number_input("How many recommendations do you want?", 1, 20, 10)

if st.button("✨ Show me my matches"):
    recs = hybrid_recommend(
        selected_movie_title=selected_movie,
        movies_df=movies,
        movie_title_to_id=movie_title_to_id,
        cf_sim_matrix=cf_sim_matrix,
        cf_inner_to_raw=cf_inner_to_raw,
        cf_raw_to_inner=cf_raw_to_inner,
        content_sim_matrix=content_sim_matrix,
        pop_stats=pop_stats,
        alpha=alpha,
        boost_popularity=boost_pop,
        n_recs=n_recs
    )
    st.success(f"Because you liked **{selected_movie}**, you might also enjoy:")
    for i, t in enumerate(recs, 1):
        st.write(f"{i}. 🎬 {t}")
