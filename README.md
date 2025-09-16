### Dataset: [Dataset](https://grouplens.org/datasets/movielens/)
### References: [Rounak Banik](https://www.kaggle.com/code/rounakbanik/movie-recommender-systems)
## Setup: 
**Needs python<=3.9 due to surprise library**
```
git clone https://github.com/vegam05/movieRecommenderSystem5
cd movieRecommenderSystem5
pip install -r requirements.txt
streamlit run app.py
```
## Usage:
1. Select a Movie

Search for a movie or choose one from the dropdown menu.

2. Choose Recommendation Type

Collaborative Filtering: Suggests movies liked by users with similar tastes.

Content-Based Filtering: Suggests movies with similar genres to the selected movie.

3. Choose Number of Recommendations

Select how many movie recommendations you want to receive.

4. Get Recommendations

Click the Recommend button.

The app will provide the selected number of movie recommendations based on your choice of movie and recommendation type.

5. How It Works (Technical Overview)

Collaborative Filtering: Uses SVD (Singular Value Decomposition) from the surprise library to predict which movies you may like based on other users’ ratings.

Content-Based Filtering: Compares movie genres using cosine similarity to recommend movies with similar content.

The hybrid system allows you to explore both approaches depending on your preference.
