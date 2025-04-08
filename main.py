# main.py
"""
Main script to run the Last.fm recommendation system pipeline.
Implements Content-Based and SVD Collaborative Filtering.
"""
import pandas as pd
import data_loader
import preprocessing
import eda
import content_based
import collaborative_filtering # Only imports SVD functions now
import evaluation
import config

def run_pipeline():
    """ Executes the main steps of the recommendation system project. """
    print("--- Starting Last.fm Music Recommendation Pipeline ---")

    # 1. Load Data
    print("\n--- 1. Loading Data ---")
    df_interactions = data_loader.load_lastfm_data()

    if df_interactions is None or df_interactions.empty:
        print("Failed to load interaction data or data is empty. Exiting.")
        return

    # 2. Feature Engineering: Create Item ID
    print("\n--- 2. Creating Item IDs ---")
    df_interactions = preprocessing.create_item_id(df_interactions)

    # 3. EDA (Optional - Run on data *before* heavy filtering/splitting)
    print("\n--- 3. Performing EDA ---")
    # Add EDA calls here if desired, e.g.:
    # eda.plot_top_artists(df_interactions)

    # 4. Preprocessing: Filtering & Splitting
    print("\n--- 4. Filtering Sparse Data ---")
    df_filtered = preprocessing.filter_sparse_data(df_interactions)
    if not df_filtered.empty:
        print(f"DEBUG main: Unique users in df_filtered: {df_filtered['user_id'].nunique()}")
    else:
        print("DEBUG main: df_filtered is empty! Check filtering/data.")
        return # Exit if filtering removed everything

    print("\n--- Splitting Data (Time-Based) ---")
    train_df, test_df = preprocessing.time_based_split(df_filtered)
    if not train_df.empty:
        print(f"DEBUG main: Unique users in train_df: {train_df['user_id'].nunique()}")
    else:
        print("DEBUG main: train_df is empty! Cannot train models.")
        # Decide whether to exit or just skip training/eval
        return
    if not test_df.empty:
        print(f"DEBUG main: Unique users in test_df: {test_df['user_id'].nunique()}")
    else:
        print("DEBUG main: test_df is empty! Evaluation will be skipped.")


    # --- Prepare components needed for different models ---
    print("\n--- Preparing Training Set Components ---")
    train_item_metadata = content_based.get_item_metadata_df(train_df)
    all_items_in_train = set(train_item_metadata['item_id'])

    # Create sparse matrix and mappings from training data (needed for SVD item map)
    # Use train_df.copy() to avoid modifying the original split dataframe
    # Note: SVD itself doesn't need the matrix, but we need the item_map from this process
    _, _, train_item_map, _ = preprocessing.create_user_item_matrix(train_df.copy())


    # 5. Train Models
    print("\n--- 5. Training Models ---")

    # Content-Based Model
    cb_model_components = None
    if not train_item_metadata.empty:
        try:
            print("Training Content-Based Model...")
            cb_feature_cols = ['artist', 'album', 'track'] # Choose features
            cb_similarity_matrix, cb_indices = content_based.build_content_similarity_matrix(train_item_metadata, feature_cols=cb_feature_cols)
            # Components needed for evaluation/prediction: sim matrix, item->index map, set of all valid items
            cb_model_components = (cb_similarity_matrix, cb_indices, all_items_in_train)
            print("Content-Based similarity matrix built.")
        except Exception as e:
            print(f"Error building Content-Based model: {e}")
    else:
        print("Skipping Content-Based model training (no metadata extracted from train set).")

    # Collaborative Filtering: SVD (Surprise)
    svd_model_components = None
    try:
        print("Training CF Model (SVD)...")
        # Pass the training dataframe directly to the Surprise trainer
        svd_algo, svd_trainset = collaborative_filtering.train_svd_model_implicit(train_df)
        # Components needed: algo, trainset (for Surprise internal IDs), item_map (for mapping back)
        svd_model_components = (svd_algo, svd_trainset, train_item_map) # Use item_map derived from train_df
        print("SVD model trained.")
    except Exception as e:
        print(f"Error training SVD model: {e}")

    # --- ALS Training Removed ---


    # 6. Evaluate Models (only if test_df is not empty)
    print("\n--- 6. Evaluating Models ---")
    if not test_df.empty:
        if svd_model_components:
            evaluation.evaluate_model(svd_model_components, test_df, train_df, model_type='svd')
        else:
            print("Skipping SVD evaluation (model not trained).")

        # --- ALS Evaluation Removed ---

        if cb_model_components:
            evaluation.evaluate_model(cb_model_components, test_df, train_df, model_type='cb')
        else:
             print("Skipping Content-Based evaluation (model not trained).")
    else:
        print("Skipping evaluation because the test set is empty.")

    # 7. Generate Example Recommendations (Optional)
    print("\n--- 7. Generating Example Recommendations ---")
    if not train_df.empty and not train_item_map.empty: # Ensure train data and map exist
        example_user_id = train_df['user_id'].iloc[0] # Pick a user from training data
        print(f"\n--- Example Recommendations for User: {example_user_id} ---")

        if svd_model_components:
             try:
                 svd_recs = collaborative_filtering.get_cf_recommendations_surprise(
                     svd_model_components[0], example_user_id, svd_model_components[1], svd_model_components[2] # algo, trainset, item_map
                 )
                 print(f"\nSVD Recommendations:")
                 print(svd_recs)
             except Exception as e:
                 print(f"Could not generate SVD recommendations: {e}")

        # --- ALS Example Recommendation Removed ---

        if cb_model_components:
             try:
                 # Get user history from the original train_df before indexing/duplicates removed
                 user_liked_items = train_df[train_df['user_id'] == example_user_id]['item_id'].unique().tolist()
                 if user_liked_items:
                     cb_recs = content_based.get_content_based_recommendations(
                         user_liked_items, cb_model_components[0], cb_model_components[1], cb_model_components[2] # sim_matrix, indices, all_items
                     )
                     print(f"\nContent-Based Recommendations (based on {len(user_liked_items)} items):")
                     print(cb_recs)
                 else:
                      print("\nUser has no history in training data for CB recommendations.")
             except Exception as e:
                  print(f"Could not generate CB recommendations: {e}")

    else:
        print("Skipping example recommendations (training data or item map is empty).")


    print("\n--- Pipeline Finished ---")

if __name__ == '__main__':
    run_pipeline()