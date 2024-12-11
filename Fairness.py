import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from collections import defaultdict
import matplotlib.pyplot as plt

""" Load MovieLens dataset"""
movies = pd.read_csv("movies.csv")  
ratings = pd.read_csv("ratings.csv")  
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

"""Analyze Genre Distribution"""
movies['genres'] = movies['genres'].str.split('|')  # Split genres
genre_counts = defaultdict(int)
for genres in movies['genres']:
    for genre in genres:
        genre_counts[genre] += 1

"""Train SVD Model and Generate Recommendations"""
trainset = data.build_full_trainset()
model = SVD()
model.fit(trainset)
testset = trainset.build_testset()
predictions = model.test(testset)

#Analyze Recommendations
movie_genres = {row['movieId']: row['genres'] for _, row in movies.iterrows()}
recommended_movies = defaultdict(list)
for pred in predictions:
    recommended_movies[pred.uid].append((pred.iid, pred.est))

recommended_genre_counts = defaultdict(int)
for user, recs in recommended_movies.items():
    for movie_id, _ in recs:
        genres = movie_genres.get(movie_id, [])
        for genre in genres:
            recommended_genre_counts[genre] += 1

# Total Recommendations
total_recommendations = sum(recommended_genre_counts.values())

"""Implement Fairness Methods"""
# Proportional Fairness
fair_recommendations_proportional = defaultdict(float)
for genre, count in recommended_genre_counts.items():
    fair_recommendations_proportional[genre] = (count / total_recommendations) * genre_counts[genre]

# Equal Exposure Fairness
total_genres = len(genre_counts)
fair_recommendations_equal = defaultdict(float)
for genre in genre_counts:
    fair_recommendations_equal[genre] = total_recommendations / total_genres

# Explicit Re-Ranking
explicit_reranked = defaultdict(int)
for user, recs in recommended_movies.items():
    genre_counts_user = defaultdict(int)
    for movie_id, _ in recs:
        genres = movie_genres.get(movie_id, [])
        for genre in genres:
            if genre_counts_user[genre] < genre_counts[genre] / sum(genre_counts.values()) * len(recs):
                explicit_reranked[genre] += 1
                genre_counts_user[genre] += 1

# Stronger Relative Re-Ranking
strong_relative_reranked = defaultdict(int)
for genre, count in recommended_genre_counts.items():
    target_count = total_recommendations * (genre_counts[genre] / sum(genre_counts.values()))
    strong_relative_reranked[genre] = min(count, target_count)

# Weighted Re-Ranking
weights = {genre: 1 / genre_counts[genre] for genre in genre_counts}
weighted_reranked = defaultdict(float)
for user, recs in recommended_movies.items():
    for movie_id, _ in recs:
        genres = movie_genres.get(movie_id, [])
        for genre in genres:
            weighted_reranked[genre] += weights.get(genre, 0)

weighted_total = sum(weighted_reranked.values())
weighted_reranked = {genre: count / weighted_total * total_recommendations for genre, count in weighted_reranked.items()}

"""Calculte RMSE"""
def calculate_rmse_for_fairness_method(predictions, adjusted_genre_counts, original_genre_counts, movie_genres):
    """Adjust predictions based on adjusted genre counts and calculate RMSE."""
    adjusted_predictions = []
    for pred in predictions:
        genres = movie_genres.get(pred.iid, [])
        adjust_factor = 1
        for genre in genres:
            if genre in adjusted_genre_counts and genre in original_genre_counts:
                adjust_factor = adjusted_genre_counts[genre] / original_genre_counts[genre]
        adjusted_rating = min(5.0, max(0.5, pred.est * adjust_factor))  # Clamp between rating scale
        adjusted_predictions.append(pred._replace(est=adjusted_rating))
    return accuracy.rmse(adjusted_predictions, verbose=False)

rmse_proportional = calculate_rmse_for_fairness_method(predictions, fair_recommendations_proportional, genre_counts, movie_genres)
rmse_equal = calculate_rmse_for_fairness_method(predictions, fair_recommendations_equal, genre_counts, movie_genres)
rmse_explicit = calculate_rmse_for_fairness_method(predictions, explicit_reranked, genre_counts, movie_genres)
rmse_strong = calculate_rmse_for_fairness_method(predictions, strong_relative_reranked, genre_counts, movie_genres)
rmse_weighted = calculate_rmse_for_fairness_method(predictions, weighted_reranked, genre_counts, movie_genres)


"""Recalculate Fairness Scores"""
def compute_fairness_scores(original, adjusted):
    """Compute fairness score as the ratio of adjusted distribution to original."""
    fairness_scores = {}
    for genre, original_count in original.items():
        adjusted_count = adjusted.get(genre, 0)
        fairness_scores[genre] = adjusted_count / original_count if original_count > 0 else 0
    return fairness_scores

fairness_proportional = compute_fairness_scores(genre_counts, fair_recommendations_proportional)
fairness_equal = compute_fairness_scores(genre_counts, fair_recommendations_equal)
fairness_explicit = compute_fairness_scores(genre_counts, explicit_reranked)
fairness_strong = compute_fairness_scores(genre_counts, strong_relative_reranked)
fairness_weighted = compute_fairness_scores(genre_counts, weighted_reranked)

"""Performance Metrics"""
performance_data = {
    'Method': ['Proportional Fairness', 'Equal Exposure', 'Explicit Re-Ranking', 'Stronger Relative Re-Ranking', 'Weighted Re-Ranking'],
    'Average Fairness Score': [
        sum(fairness_proportional.values()) / len(fairness_proportional),
        sum(fairness_equal.values()) / len(fairness_equal),
        sum(fairness_explicit.values()) / len(fairness_explicit),
        sum(fairness_strong.values()) / len(fairness_strong),
        sum(fairness_weighted.values()) / len(fairness_weighted)
    ],
    'Maximum Fairness Score': [
        max(fairness_proportional.values()),
        max(fairness_equal.values()),
        max(fairness_explicit.values()),
        max(fairness_strong.values()),
        max(fairness_weighted.values())
    ],
    'Minimum Fairness Score': [
        min(fairness_proportional.values()),
        min(fairness_equal.values()),
        min(fairness_explicit.values()),
        min(fairness_strong.values()),
        min(fairness_weighted.values())
    ],
    'RMSE': [rmse_proportional,
    rmse_equal ,
    rmse_explicit,
    rmse_strong ,
    rmse_weighted]
        
}

performance_df = pd.DataFrame(performance_data)

"""Visualization"""
def plot_all_methods(original, recommended, proportional, equal, explicit, strong, weighted, title):
    genres = list(original.keys())
    x = np.arange(len(genres))

    plt.figure(figsize=(15, 8))
    plt.bar(x - 0.4, [original[genre] for genre in genres], width=0.1, label='Original', color='blue')
    plt.bar(x - 0.3, [recommended.get(genre, 0) for genre in genres], width=0.1, label='Recommended', color='orange')
    plt.bar(x - 0.2, [proportional.get(genre, 0) for genre in genres], width=0.1, label='Proportional', color='green')
    plt.bar(x - 0.1, [equal.get(genre, 0) for genre in genres], width=0.1, label='Equal', color='red')
    plt.bar(x, [explicit.get(genre, 0) for genre in genres], width=0.1, label='Explicit Re-Rank', color='purple')
    plt.bar(x + 0.1, [strong.get(genre, 0) for genre in genres], width=0.1, label='Strong Re-Rank', color='brown')
    plt.bar(x + 0.2, [weighted.get(genre, 0) for genre in genres], width=0.1, label='Weighted Re-Rank', color='pink')

    plt.xlabel('Genre')
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(x, genres, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_all_methods(
    genre_counts, recommended_genre_counts, fair_recommendations_proportional, fair_recommendations_equal,
    explicit_reranked, strong_relative_reranked, weighted_reranked,
    "Genre Distribution with Different Fairness Methods"
)

print(performance_df)
