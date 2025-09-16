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
Just search for a movie or choose one from the dropdown menu and hit the Recommend button, you'll be provided with 10 movie names based on the data acquired from the dataset that imitates the taste of other users who liked the input movie.

The system functions upon collaborative filtering and hence it functions based on the idea that- users' ratings similar to me can be used to predict how much I will like a particular product or service, those users have used/experienced but I have not.
The system is implemented using SVD from the surprise library
