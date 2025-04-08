# collaborative_filtering.py
"""
Implementation of Collaborative Filtering using Matrix Factorization (SVD from Surprise).
Requires 'surprise': pip install scikit-surprise
"""
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_split
import config
# Removed import create_user_item_matrix (if only used for ALS example previously)
# Removed implicit library imports and checks

# --- SVD Example (using Surprise) ---

def train_svd_model_implicit(df_interactions, user_col='user_id', item_col='item_id'):
    """
    Trains an SVD model using Surprise library on implicit binary data (listens = 1).
    """
    print("Training SVD model on implicit binary data...")

    # Data needs user, item, and a "rating". We'll add a constant rating of 1.
    df_surprise = df_interactions[[user_col, item_col]].copy()
    df_surprise['rating'] = 1 # Treat every listen event as a rating of 1

    # Define the rating scale. Since it's binary, it's just (1, 1).
    reader = Reader(rating_scale=(1, 1))

    # Load data into Surprise dataset format
    data = Dataset.load_from_df(df_surprise, reader)

    # Build the full training set (use pre-split data if available)
    trainset = data.build_full_trainset()
    print(f"Surprise trainset built with {trainset.n_users} users and {trainset.n_items} items.")

    # Initialize and train the SVD algorithm
    # TODO: Tune parameters if needed
    algo = SVD(n_factors=config.N_LATENT_FACTORS_SVD, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    algo.fit(trainset)

    print("SVD model training complete.")
    return algo, trainset

def get_cf_recommendations_surprise(algo, user_id, trainset, item_map, top_n=config.N_RECOMMENDATIONS):
    """
    Generates recommendations for a user using a trained Surprise model (like SVD).
    Needs item_map to translate internal Surprise item IDs back to original item_ids.
    """
    # Convert external user_id to internal Surprise user id (integer)
    try:
        user_inner_id = trainset.to_inner_uid(user_id)
    except ValueError:
        # print(f"User '{user_id}' not found in the training set.")
        return [] # Return empty list if user is unknown to the model

    # Get all item *inner* ids known to the trainset
    all_item_inner_ids = list(trainset.all_items())

    # Get items the user *has* interacted with in the trainset (as inner ids)
    user_interacted_items_inner = {item_inner_id for (item_inner_id, _) in trainset.ur[user_inner_id]}

    # Prepare list of items to predict (those the user hasn't interacted with)
    items_to_predict_inner = [iid for iid in all_item_inner_ids if iid not in user_interacted_items_inner]

    # Create testset tuples (raw_uid, raw_iid, dummy_rating)
    # Need to map inner item ID back to raw item ID using trainset.to_raw_iid()
    testset_tuples = [(user_id, trainset.to_raw_iid(item_inner_id), 1.0) for item_inner_id in items_to_predict_inner]

    if not testset_tuples:
        # print(f"User {user_id} has interacted with all items or no items left to predict.")
        return []

    # Make predictions
    predictions = algo.test(testset_tuples)

    # Sort predictions by estimated rating (higher is better)
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get top N recommended *original* item_ids
    recommended_items = [pred.iid for pred in predictions[:top_n]]

    return recommended_items

# --- ALS Code Removed ---


if __name__ == '__main__':
    # Simplified example focusing only on SVD
    from data_loader import load_lastfm_data
    from preprocessing import create_item_id, filter_sparse_data, time_based_split, create_user_item_matrix # Need maps

    df_inter = load_lastfm_data()
    if df_inter is not None:
        df_with_id = create_item_id(df_inter)
        df_filtered = filter_sparse_data(df_with_id)
        train_df, test_df = time_based_split(df_filtered)

        if train_df is not None and not train_df.empty:
            print("\n--- Running SVD Example ---")
            try:
                 # Train SVD model on the training data
                 svd_model, svd_trainset = train_svd_model_implicit(train_df)

                 # Need item_map from the training data context
                 # This requires running matrix creation on train_df to get the map
                 _, _, svd_item_map, _ = create_user_item_matrix(train_df.copy()) # Get item map specific to train set

                 # Get recommendations for an example user from the train set
                 if not svd_item_map.empty: # Check if item map was created
                      example_user_svd = train_df['user_id'].iloc[0]
                      recommendations_svd = get_cf_recommendations_surprise(svd_model, example_user_svd, svd_trainset, svd_item_map)
                      print(f"SVD Recommendations for user '{example_user_svd}':")
                      print(recommendations_svd)
                 else:
                      print("Could not generate SVD recommendations: item map is empty.")

            except Exception as e:
                 print(f"Could not run SVD example: {e}")

        else:
            print("Training data is empty, cannot run CF examples.")

    # ALS Example section removed