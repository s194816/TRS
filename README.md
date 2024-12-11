
# Fairness and Robustness in Recommendation Systems

This project explores fairness and robustness in recommendation systems. It implements and evaluates algorithms related to fairness in recommendations and investigates robustness to fake data.


## Contents

- **Fairness.py**: Implements fairness-aware methods for movie recommendation systems, using the MovieLens dataset. It measures fairness and recommendation quality metrics and visualizes genre distributions.
- **Robustness_Fake_Data.py**: Explores the robustness of recommendation systems to fake data. It trains models on datasets with random and targeted fake data and analyzes the impact on performance.

## Requirements

- Python 3.8 or later
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `surprise`
  -  `collections`

## Setup

1. Install dependencies:  
   ```bash
   pip install pandas numpy matplotlib scikit-surprise
   ```
2. The required datasets:
   - `Fairness.py` expects `movies.csv` and `ratings.csv` (from MovieLens dataset). These are included in zip-file.
   - `Robustness_Fake_Data.py` uses the built-in `ml-100k` dataset from `surprise`.

## Fairness Analysis

`Fairness.py` implements various fairness-aware algorithms:
- **Proportional Fairness**
- **Equal Exposure Fairness**
- **Explicit and Strong Re-Ranking**
- **Weighted Re-Ranking**

### Outputs:
- **Fairness Metrics**: Average, maximum, and minimum fairness scores for each method.
- **RMSE Metrics**: Evaluates the impact of fairness adjustments on recommendation quality.
- **Visualization**: Plots genre distributions for each fairness method.

## Robustness Testing

`Robustness_Fake_Data.py` examines how recommendation systems respond to fake data.
- **Random Fake Data**: Simulates extreme ratings for random movies.
- **Targeted Fake Data**: Adds high ratings to specific movies to simulate manipulation.

### Methods:
- **Models**: SVD, KNNBasic, BaselineOnly.
- **Metrics**: RMSE, MAE, and detection rate of fake data.

### Outputs:
- **Reconstructed Datasets**: Filters out fake data using predictions.
- **Visualization**: Compares ratings distributions for original, fake, and corrected data.

## Usage

1. Run `Fairness.py`:
   ```bash
   python Fairness.py
   ```
   Outputs: Fairness metrics table and genre distribution plots.

2. Run `Robustness_Fake_Data.py`:
   ```bash
   python Robustness_Fake_Data.py
   ```
   Outputs: RMSE, fake data detection rates, and ratings distribution plots.

