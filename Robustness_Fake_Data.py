import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, KNNBasic, BaselineOnly
from surprise.model_selection import train_test_split
from surprise import accuracy
import matplotlib.pyplot as plt
from collections import defaultdict

"""Load the MovieLens dataset"""
data = Dataset.load_builtin('ml-100k')
reader = Reader(rating_scale=(0.5, 5))

"""Inject fake data"""
def inject_fake_data(raw_ratings, num_fake_users=100, num_fake_ratings_per_user=20, targeted=False):
    max_user_id = int(max(raw_ratings, key=lambda x: int(x[0]))[0])
    max_movie_id = int(max(raw_ratings, key=lambda x: int(x[1]))[1])

    fake_ratings = []
    for i in range(num_fake_users):
        fake_user_id = max_user_id + i + 1
        for j in range(num_fake_ratings_per_user):
            if targeted:
                fake_movie_id = np.random.randint(1, 51)  # Targeted to first 50 movies
                fake_rating = 5.0  # High ratings for targeted fake data
            else:
                fake_movie_id = np.random.randint(1, max_movie_id + 1)
                fake_rating = np.random.choice([0.5, 5])  # Extreme ratings for random noise
            fake_ratings.append((str(fake_user_id), str(fake_movie_id), fake_rating, None))
    return raw_ratings + fake_ratings

raw_ratings = list(data.raw_ratings)
original_user_item_pairs = {(uid, iid) for uid, iid, _, _ in raw_ratings}
raw_ratings_with_random_fake = inject_fake_data(raw_ratings)
raw_ratings_with_targeted_fake = inject_fake_data(raw_ratings, targeted=True)

data_with_random_fake = Dataset.load_from_df(
    pd.DataFrame([(u, i, r) for u, i, r, _ in raw_ratings_with_random_fake], columns=['userId', 'movieId', 'rating']),
    reader
)
data_with_targeted_fake = Dataset.load_from_df(
    pd.DataFrame([(u, i, r) for u, i, r, _ in raw_ratings_with_targeted_fake], columns=['userId', 'movieId', 'rating']),
    reader
)

""" Reconstruction function"""
def reconstruct_data_from_predictions(data_fake, predictions, original_pairs):
    reconstructed_ratings = defaultdict(float)
    for pred in predictions:
        uid = str(pred.uid)
        iid = str(pred.iid)
        if (uid, iid) in original_pairs:  # Only retain original user-item pairs
            reconstructed_ratings[(uid, iid)] = pred.est

    reconstructed_raw_ratings = [
        (uid, iid, est_rating) for (uid, iid), est_rating in reconstructed_ratings.items()
    ]
    reconstructed_data = Dataset.load_from_df(
        pd.DataFrame(reconstructed_raw_ratings, columns=['userId', 'movieId', 'rating']), reader
    )
    return reconstructed_data



"""Train and evaluate models """
def train_and_reconstruct(data_fake, model, model_name, original_pairs):
    print(f"\n--- Training {model_name} ---")
    trainset = data_fake.build_full_trainset()
    model.fit(trainset)
    
    # Evaluate on the training set
    predictions = model.test(trainset.build_testset())
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    print(f"RMSE for {model_name}: {rmse}")
    print(f"MAE for {model_name}: {mae}")

    reconstructed_data = reconstruct_data_from_predictions(data_fake, predictions, original_pairs)

    # Analyze detection of fake data
    fake_detected = sum(1 for pred in predictions if (str(pred.uid), str(pred.iid)) not in original_pairs)
    total_fake = len(predictions) - len(original_pairs)
    print(f"Fake data detected by {model_name}: {fake_detected} / {total_fake} ({(fake_detected / total_fake) * 100:.2f}%)")
    
    return reconstructed_data

models = {
    "SVD": SVD(),
    "KNN": KNNBasic(sim_options={'user_based': True}),
    "BaselineOnly": BaselineOnly(),
}

methods_results_random = {}
methods_results_targeted = {}

for model_name, model in models.items():
    print(f"Training {model_name} on random fake data...")
    methods_results_random[model_name] = train_and_reconstruct(data_with_random_fake, model, model_name, original_user_item_pairs)
    
    print(f"Training {model_name} on targeted fake data...")
    methods_results_targeted[model_name] = train_and_reconstruct(data_with_targeted_fake, model, model_name, original_user_item_pairs)

"""Visualization and some analysis"""
def compute_rating_counts(data):
    counts = defaultdict(int)
    for (_, _, rating, _) in data.raw_ratings:
        rounded_rating = int(round(rating))  # Round to nearest integer
        counts[rounded_rating] += 1
    return counts

def bar_and_line_plot_ratings_distribution(original_data, fake_data, corrected_results, title):
    original_counts = compute_rating_counts(original_data)
    fake_counts = compute_rating_counts(fake_data)
    corrected_counts = {method: compute_rating_counts(result) for method, result in corrected_results.items()}

    x_labels = sorted(original_counts.keys())
    width = 0.35  # Width for the bar plots

    plt.figure(figsize=(12, 6))
    
    # Bar plots for original and fake data
    plt.bar([x - width/2 for x in x_labels], [original_counts[x] for x in x_labels], width, label='Original Data', color='blue')
    plt.bar([x + width/2 for x in x_labels], [fake_counts[x] for x in x_labels], width, label='Fake Data', color='orange')

    # plots for corrected results
    for method, counts in corrected_counts.items():
        plt.plot(x_labels, [counts[x] for x in x_labels], marker='o', label=f'{method} Corrected')

    plt.xlabel("Rating")
    plt.ylabel("Number of Ratings")
    plt.title(title)
    plt.legend()
    plt.show()

print("Plotting random fake data results...")
bar_and_line_plot_ratings_distribution(data, data_with_random_fake, methods_results_random, "Ratings Distribution: Random Noise Fake Data")

print("Plotting targeted fake data results...")
bar_and_line_plot_ratings_distribution(data, data_with_targeted_fake, methods_results_targeted, "Ratings Distribution: Targeted Fake Data")
