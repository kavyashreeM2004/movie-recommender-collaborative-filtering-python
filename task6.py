import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample user-item rating data
data = {'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4],
        'movie_id': [10, 20, 30, 10, 30, 20, 30, 40, 10],
        'rating': [4, 5, 2, 5, 3, 4, 4, 5, 2]}

df = pd.DataFrame(data)

# Create user-item matrix (pivot table)
user_item_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Function to get movie recommendations for a given user
def recommend_movies(user_id, user_item_matrix, user_similarity_df, top_n=3):
    # Get similar users sorted by similarity score (excluding the user itself)
    similar_users = user_similarity_df[user_id].drop(user_id).sort_values(ascending=False)
    
    # Weighted average of ratings from similar users for each movie
    weighted_ratings = user_item_matrix.loc[similar_users.index].T.dot(similar_users) / similar_users.sum()
    
    # Remove movies already rated by the user
    user_rated_movies = user_item_matrix.loc[user_id]
    recommendations = weighted_ratings[user_rated_movies == 0]
    
    # Recommend top N movies with highest predicted rating
    recommended_movies = recommendations.sort_values(ascending=False).head(top_n)
    return recommended_movies

# Example: Get recommendations for user 1
recommendations = recommend_movies(user_id=1, user_item_matrix=user_item_matrix, user_similarity_df=user_similarity_df, top_n=3)
print("Recommended movies for user 1:")
print(recommendations)
