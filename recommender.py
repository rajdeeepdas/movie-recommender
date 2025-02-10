import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load MovieLens dataset (same as LightFM)
url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
columns = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv(url, sep="\t", names=columns)

# Drop timestamp column as it's not needed
df = df.drop("timestamp", axis=1)

# Define the rating scale
reader = Reader(rating_scale=(1, 5))

# Load data into Surprise format
data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Create model (SVD for collaborative filtering)
model = SVD()
model.fit(trainset)

# Make predictions
predictions = model.test(testset)

# Compute RMSE (Root Mean Squared Error)
rmse = accuracy.rmse(predictions)
print(f"Model RMSE: {rmse:.4f}")

# Recommend movies for specific users
def recommend_movies(model, user_ids, n_recommendations=3):
    unique_items = df["item_id"].unique()
    
    for user_id in user_ids:
        print(f"\nUser {user_id} Recommendations:")

        # Predict ratings for all items the user hasn't rated
        user_predictions = [
            (item_id, model.predict(user_id, item_id).est)
            for item_id in unique_items
        ]

        # Sort items by predicted rating (highest first)
        user_predictions.sort(key=lambda x: x[1], reverse=True)

        # Print top recommended items
        for item_id, predicted_rating in user_predictions[:n_recommendations]:
            print(f"  Movie ID: {item_id}, Predicted Rating: {predicted_rating:.2f}")

# Recommend movies for users 3, 25, 450
recommend_movies(model, [3, 25, 450])
