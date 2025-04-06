# config.py
"""
Configuration settings for the Last.fm recommendation project.
"""

# --- Data Paths ---
# *** UPDATE THIS PATH to your actual Last.fm dataset file ***
LASTFM_DATA_PATH = './lastfm_data.csv' # Example path - CHANGE THIS

# --- Preprocessing ---
MIN_INTERACTIONS_PER_USER = 5  # Min listens per user
MIN_INTERACTIONS_PER_ITEM = 5  # Min listens per track (Artist - Track)
ITEM_ID_SEPARATOR = ' - ' # Separator for creating item_id from Artist and Track

# --- Model Settings ---
N_LATENT_FACTORS_SVD = 50 # Example for Matrix Factorization (SVD)
N_RECOMMENDATIONS = 10   # Number of recommendations to generate
# For implicit ALS (if used)
N_LATENT_FACTORS_ALS = 50
REGULARIZATION_ALS = 0.01
ITERATIONS_ALS = 15


# --- Evaluation ---
TEST_SET_SIZE = 0.2 # Proportion of data for testing (using time-based split logic)