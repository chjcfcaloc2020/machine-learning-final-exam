import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

movies_df = pd.read_csv("D:/VKU_2_3/Shiny/movies.csv", usecols=['movieId', 'title'])
rating_df = pd.read_csv("D:/VKU_2_3/Shiny/ratings.csv", usecols=['userId', 'movieId', 'rating'], 
            dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})

df = pd.merge(rating_df, movies_df, on="movieId")
# print(df.head())

combine_movie_rating = df.dropna(axis=0, subset=['title'])
movie_rating_count = (combine_movie_rating.groupby(by=['title'])['rating'].count().reset_index().
                      rename(columns={'rating': 'totalRatingCount'})[['title', 'totalRatingCount']])
# print(movie_rating_count.head())

rating_with_totalRatingCount = combine_movie_rating.merge(movie_rating_count, left_on='title', right_on='title', how='left')
# print(rating_with_totalRatingCount.head())

pd.set_option('display.float_format', lambda x: '%.3f' % x)
# print(movie_rating_count['totalRatingCount'].describe())

popularity_threshold = 50
rating_popular_movie = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
# print(rating_popular_movie.head())
# print(rating_popular_movie.shape)

movie_features_df = rating_popular_movie.pivot_table(index='title', columns='userId', values='rating').fillna(0)
# print(movie_features_df.head())

movie_features_df_matrix = csr_matrix(movie_features_df.values)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(movie_features_df_matrix)
# print(movie_features_df.shape)

query_index = np.random.choice(movie_features_df.shape[0])
# print(query_index)
distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors=6)
# print(movie_features_df.head())

for i in range(0, len(distances.flatten())):
    if i == 0:
        print("Recommendations for {0}:\n".format(movie_features_df.index[query_index]))
    else:
        print("{0}: {1}, with distance of {2}:".format(i, movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))