import pandas as pd
import numpy as np

# Create a mock dataset with user ratings for movies
data_dict = {
    'user': ['Alice', 'Alice', 'Alice', 'Bob', 'Bob', 'Charlie', 'David', 'David', 'Eve', 'Eve'],
    'movie': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
              'Pulp Fiction', 'The Lord of the Rings: The Return of the King',
              'Forrest Gump', 'Inception', 'Fight Club', 
              'The Matrix', 'Gladiator'],
    'rating': [5, 4, 0, 5, 0, 3, 0, 4, 5, 2]
}

# Convert to a pandas DataFrame
df = pd.DataFrame(data_dict)

# Create a user-item interaction matrix
user_item_matrix = df.pivot_table(index='user', columns='movie', values='rating').fillna(0)

# Calculate cosine similarity between users
def cosine_similarity(matrix):
    sim_matrix = np.dot(matrix, matrix.T)
    norms = np.array([np.sqrt(np.diagonal(sim_matrix))])
    return sim_matrix / norms / norms.T

user_similarities = cosine_similarity(user_item_matrix.values)

# Create a DataFrame for user similarities
user_sim_df = pd.DataFrame(user_similarities, index=user_item_matrix.index, columns=user_item_matrix.index)

# Create a function to get movie recommendations for a user
def get_movie_recommendations(user, num_recommendations=3):
    # Get the user's ratings
    user_ratings = user_item_matrix.loc[user]
    
    # Find similar users
    similar_users = user_sim_df[user].sort_values(ascending=False).index[1:]  # Exclude the user themselves
    
    recommendations = {}
    
    for similar_user in similar_users:
        similar_user_ratings = user_item_matrix.loc[similar_user]
        for movie, rating in similar_user_ratings.items():
            if user_ratings[movie] == 0 and rating > 0:  # If the user hasn't rated this movie and the similar user has
                if movie not in recommendations:
                    recommendations[movie] = rating
                else:
                    recommendations[movie] += rating  # Aggregate the scores
    
    # Sort recommendations based on scores
    recommended_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return recommended_movies[:num_recommendations]

# Console application
def main():
    print("Welcome to the Movie Recommendation System!")
    user = input("Enter your name: ")

    # Get movie recommendations for the user
    recommendations = get_movie_recommendations(user)
    
    print(f"\nRecommended movies for {user}:")
    if recommendations:
        for movie, score in recommendations:
            print(f"{movie} (Score: {score})")
    else:
        print("No recommendations available.")

if __name__ == "__main__":
    main()
