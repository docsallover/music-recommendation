# collaborative_filtering.py
"""
Implementation of Collaborative Filtering.
Example using SVD from Surprise for implicit binary data.
Mentions ALS from 'implicit' library as a good alternative.

Requires 'surprise': pip install scikit-surprise
Optional 'implicit': pip install implicit (+ BLAS setup may be needed)
"""
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_split
import config
from preprocessing import create_user_item_matrix # To get user/item maps if needed elsewhere

try:
    import threadpoolctl
    threadpoolctl.threadpool_limits(1, "blas")
except Exception as e:
    print(f"Could not set threadpool limits: {e}")


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
        print(f"User {user_id} has interacted with all items or no items left to predict.")
        return []

    # Make predictions
    predictions = algo.test(testset_tuples)

    # Sort predictions by estimated rating (higher is better)
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get top N recommended *original* item_ids
    recommended_items = [pred.iid for pred in predictions[:top_n]]

    return recommended_items

# --- ALS Example (using implicit library - OPTIONAL but recommended for implicit data) ---
try:
    import implicit
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False
    print("Warning: 'implicit' library not found. Skipping ALS example functions.")
    print("Install with: pip install implicit")

def train_als_model(sparse_user_item_matrix):
    """ Trains an ALS model using the implicit library. """
    if not IMPLICIT_AVAILABLE: return None
    print("Training ALS model using 'implicit' library...")
    # Initialize the model
    # TODO: Tune factors, regularization, iterations
    model = implicit.als.AlternatingLeastSquares(
        factors=config.N_LATENT_FACTORS_ALS,
        regularization=config.REGULARIZATION_ALS,
        iterations=config.ITERATIONS_ALS,
        random_state=42
    )

    # Train the model. Requires items x users matrix.
    # Our sparse_matrix is users x items, so transpose it.
    item_user_matrix = sparse_user_item_matrix.T.tocsr()
    model.fit(item_user_matrix)

    print("ALS model training complete.")
    return model

def get_als_recommendations(model, user_idx, sparse_user_item_matrix, item_map, top_n=config.N_RECOMMENDATIONS):
    """ Generates recommendations using a trained implicit ALS model. """
    if not IMPLICIT_AVAILABLE or model is None: return []

    # Get recommendations (recommend expects user_idx, user_item matrix)
    # N = top_n + number of items user already interacted with (approx) to ensure we get enough *new* items
    # A safer way is to request more items and filter afterwards.
    n_to_request = top_n + int(sparse_user_item_matrix[user_idx].nnz * 1.5) # Heuristic
    if n_to_request < top_n * 2: n_to_request = top_n * 2 # Ensure requesting a decent amount


    # recommended: List of (item_idx, score) tuples
    recommended_idxs_scores = model.recommend(user_idx, sparse_user_item_matrix[user_idx], N=n_to_request, filter_already_liked_items=True)

    # Map item_idx back to original item_id using item_map
    recommended_items = []
    for item_idx, score in recommended_idxs_scores:
         if item_idx in item_map.index: # Check if index exists in map
             recommended_items.append(item_map.loc[item_idx, 'item_id'])
         if len(recommended_items) >= top_n:
             break

    return recommended_items


if __name__ == '__main__':
    from data_loader import load_lastfm_data
    from preprocessing import create_item_id, filter_sparse_data, time_based_split, create_user_item_matrix

    df_inter = load_lastfm_data()
    if df_inter is not None:
        df_with_id = create_item_id(df_inter)
        df_filtered = filter_sparse_data(df_with_id)
        train_df, test_df = time_based_split(df_filtered)

        if train_df is not None and not train_df.empty:
            # --- SVD Example ---
            print("\n--- Running SVD Example ---")
            try:
                 # Train SVD model on the training data
                 svd_model, svd_trainset = train_svd_model_implicit(train_df)

                 # We need item_map from the *training* data context for Surprise prediction mapping
                 # Recreate matrix and maps based *only* on train_df for this context
                 _, _, svd_item_map, _ = create_user_item_matrix(train_df.copy())

                 # Get recommendations for an example user from the train set
                 example_user_svd = train_df['user_id'].iloc[0]
                 recommendations_svd = get_cf_recommendations_surprise(svd_model, example_user_svd, svd_trainset, svd_item_map)
                 print(f"SVD Recommendations for user '{example_user_svd}':")
                 print(recommendations_svd)
            except Exception as e:
                 print(f"Could not run SVD example: {e}")


            # --- ALS Example (Optional) ---
            if IMPLICIT_AVAILABLE:
                 print("\n--- Running ALS Example ---")
                 try:
                     # Create sparse matrix and maps from training data
                     train_sparse_matrix, train_user_map, train_item_map, train_data_indexed = create_user_item_matrix(train_df.copy())

                     als_model = train_als_model(train_sparse_matrix)

                     if als_model:
                         # Get an example user's index
                         example_user_als = train_user_map['user_id'].iloc[0]
                         example_user_idx = train_user_map[train_user_map['user_id'] == example_user_als].index[0]

                         recommendations_als = get_als_recommendations(als_model, example_user_idx, train_sparse_matrix, train_item_map)
                         print(f"ALS Recommendations for user '{example_user_als}' (index {example_user_idx}):")
                         print(recommendations_als)
                 except Exception as e:
                     print(f"Could not run ALS example: {e}")
        else:
            print("Training data is empty, cannot run CF examples.")