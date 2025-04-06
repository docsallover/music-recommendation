# collaborative_filtering.py
"""
Implementation of Collaborative Filtering using Matrix Factorization (e.g., SVD).
Requires the 'surprise' library: pip install scikit-surprise
"""
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_split # Avoids conflict
import config
from preprocessing import create_user_item_matrix # To potentially get user/item maps

def train_svd_model(df_interactions, user_col='user_id', item_col='song_id', rating_col='interaction_count'):
    """ Trains an SVD model using the Surprise library. """
    print("Training SVD model...")
    if rating_col not in df_interactions.columns:
         # If no explicit rating, maybe create one (e.g., binary 1 for interaction)
         # Or use models designed for implicit feedback (like ALS from 'implicit' library)
         print(f"Warning: Rating column '{rating_col}' not found. Using binary interaction (1).")
         df_interactions = df_interactions.copy() # Avoid modifying original df
         df_interactions['binary_rating'] = 1
         rating_col = 'binary_rating'


    # Define the rating scale (adjust if necessary, esp. if using play counts directly)
    # Determine min/max from the data or set reasonable defaults
    min_rating = df_interactions[rating_col].min()
    max_rating = df_interactions[rating_col].max()
    reader = Reader(rating_scale=(min_rating, max_rating))

    # Load data into Surprise dataset format
    data = Dataset.load_from_df(df_interactions[[user_col, item_col, rating_col]], reader)

    # Split data within Surprise (optional, can also use pre-split data)
    # trainset, testset = surprise_split(data, test_size=0.2)
    # Or build full trainset if using pre-split data
    trainset = data.build_full_trainset()


    # Initialize and train the SVD algorithm
    # TODO: Tune parameters like n_factors, n_epochs, lr_all, reg_all
    algo = SVD(n_factors=config.N_LATENT_FACTORS_SVD, random_state=42)
    algo.fit(trainset)

    print("SVD model training complete.")
    return algo, trainset

def get_cf_recommendations(algo, user_id, trainset, item_map, top_n=config.N_RECOMMENDATIONS):
    """ Generates recommendations for a user using a trained Surprise model. """

    # Convert external user_id to internal Surprise user id (integer)
    try:
        user_inner_id = trainset.to_inner_uid(user_id)
    except ValueError:
        print(f"User '{user_id}' not found in the training set.")
        return [] # Or return popular items

    # Get all item inner ids
    all_item_inner_ids = list(trainset.all_items())

    # Predict ratings for all items the user hasn't interacted with
    items_to_predict = []
    user_interacted_items = {item_inner_id for (item_inner_id, _) in trainset.ur[user_inner_id]}

    for item_inner_id in all_item_inner_ids:
        if item_inner_id not in user_interacted_items:
             # Convert inner item id back to original item_id from the map
             original_item_id = item_map.loc[trainset.to_raw_iid(item_inner_id), 'song_id']
             items_to_predict.append((user_id, original_item_id, 0)) # 0 is a placeholder rating

    if not items_to_predict:
        print(f"User {user_id} has interacted with all items in the training set.")
        return []

    # Make predictions
    predictions = algo.test([(uid, iid, r) for uid, iid, r in items_to_predict])

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get top N recommendations
    recommended_songs = [pred.iid for pred in predictions[:top_n]]

    return recommended_songs


if __name__ == '__main__':
    from data_loader import load_interaction_data
    from preprocessing import filter_sparse_data, time_based_split

    df_inter = load_interaction_data()
    if df_inter is not None:
        df_filt = filter_sparse_data(df_inter)

        # Need user/item maps - create from the *entire* filtered dataset
        # before splitting to ensure consistent mapping.
        # Note: This is slightly simplified; ideally, maps handle unseen items in test too.
        _, user_mapping, item_mapping, df_filt_indexed = create_user_item_matrix(df_filt.copy())


        if 'timestamp' in df_filt_indexed.columns:
             train_df, test_df = time_based_split(df_filt_indexed)
        else:
             # Handle case without timestamp (e.g., random split, but less ideal)
             print("Warning: Timestamp column not found. Using random split for CF example.")
             # This requires careful handling of indices if using Surprise's split
             # For simplicity here, we'll just train on the full filtered set
             train_df = df_filt_indexed


        # Decide which interaction column to use (e.g., 'interaction_count' or create 'binary_rating')
        # Let's assume 'interaction_count' exists for this example
        rating_c = 'interaction_count' if 'interaction_count' in train_df.columns else None
        if rating_c is None:
            print("Error: No suitable rating column found for SVD.")
        else:
            # Train the model on the training portion
            svd_model, training_set = train_svd_model(train_df, rating_col=rating_c)

            # Example: Get recommendations for a user (replace with actual user ID)
            example_user = train_df['user_id'].iloc[0] # Get first user_id from train set

            recommendations = get_cf_recommendations(svd_model, example_user, training_set, item_mapping)

            print(f"\nCollaborative Filtering Recommendations for user '{example_user}':")
            print(recommendations)