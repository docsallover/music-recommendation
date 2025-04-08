# config.py
"""
Configuration settings for the Last.fm recommendation project.
"""

# --- Data Paths ---
# *** UPDATE THIS PATH to your actual Last.fm dataset file ***
LASTFM_DATA_PATH = './lastfm_data.csv' # Example path - CHANGE THIS

# --- Preprocessing ---
# *** Consider adjusting these based on previous runs to get more users ***
MIN_INTERACTIONS_PER_USER = 5 # Min listens per user (Example: lowered)
MIN_INTERACTIONS_PER_ITEM = 5 # Min listens per track (Example: lowered)
ITEM_ID_SEPARATOR = ' - ' # Separator for creating item_id from Artist and Track

# --- Model Settings ---
N_LATENT_FACTORS_SVD = 50 # Example for Matrix Factorization (SVD)
N_RECOMMENDATIONS = 10   # Number of recommendations to generate
# ALS parameters removed

# --- Evaluation ---
TEST_SET_SIZE = 0.2 # Proportion of data for testing (using time-based split logic)