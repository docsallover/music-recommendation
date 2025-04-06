# main.py
"""
Main script to run the Last.fm recommendation system pipeline.
"""
import pandas as pd
import data_loader
import preprocessing
import eda
import content_based
import collaborative_filtering # Imports SVD and ALS functions/checks
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
    # eda.plot_top_artists(df_interactions)
    # eda.plot_top_tracks(df_interactions)
    # eda.plot_user_activity(df_interactions)
    # Consider sampling for EDA if dataset is very large

    # 4. Preprocessing: Filtering & Splitting
    print("\n--- 4. Filtering Sparse Data ---")
    df_filtered = preprocessing.filter_sparse_data(df_interactions)

    if df_filtered.empty:
        print("Data is empty after filtering. Cannot proceed. Check filtering thresholds.")
        return

    print("\n--- Splitting Data (Time-Based) ---")
    train_df, test_df = preprocessing.time_based_split(df_filtered)

    if train_df.empty:
        print("Training data is empty after split. Cannot train models.")
        return
    if test_df.empty:
        print("Warning: Test data is empty after split. Evaluation will not run.")


    # --- Prepare components needed for different models ---
    # Unique items from training data for CB and potentially mappings
    print("\n--- Preparing Training Set Components ---")
    train_item_metadata = content_based.get_item_metadata_df(train_df)
    all_items_in_train = set(train_item_metadata['item_id'])

    # Sparse matrix and mappings from training data (for ALS and potentially evaluation)
    # Use train_df.copy() to avoid modifying the original split dataframe
    train_sparse_matrix, train_user_map, train_item_map, train_df_indexed = preprocessing.create_user_item_matrix(train_df.copy())


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

    # Collaborative Filtering: ALS (implicit) - Optional but recommended
    als_model_components = None
    if collaborative_filtering.IMPLICIT_AVAILABLE:
         try:
             print("Training CF Model (ALS)...")
             # ALS uses the sparse matrix directly
             als_model = collaborative_filtering.train_als_model(train_sparse_matrix)
             if als_model:
                  # Components needed: model, sparse matrix, user map, item map
                  als_model_components = (als_model, train_sparse_matrix, train_user_map, train_item_map)
                  print("ALS model trained.")
         except Exception as e:
             print(f"Error training ALS model: {e}")
    else:
         print("Skipping ALS model training ('implicit' library not installed or import failed).")


    # 6. Evaluate Models (only if test_df is not empty)
    print("\n--- 6. Evaluating Models ---")
    if not test_df.empty:
        if svd_model_components:
            evaluation.evaluate_model(svd_model_components, test_df, train_df, model_type='svd')
        else:
            print("Skipping SVD evaluation (model not trained).")

        if als_model_components:
             evaluation.evaluate_model(als_model_components, test_df, train_df, model_type='als')
        else:
             print("Skipping ALS evaluation (model not trained or library unavailable).")

        if cb_model_components:
            evaluation.evaluate_model(cb_model_components, test_df, train_df, model_type='cb')
        else:
             print("Skipping Content-Based evaluation (model not trained).")
    else:
        print("Skipping evaluation because the test set is empty.")

    # 7. Generate Example Recommendations (Optional)
    print("\n--- 7. Generating Example Recommendations ---")
    if not train_df.empty:
        example_user_id = train_user_map['user_id'].iloc[0] # Pick a user from training map
        print(f"\n--- Example Recommendations for User: {example_user_id} ---")

        if svd_model_components:
             try:
                 svd_recs = collaborative_filtering.get_cf_recommendations_surprise(
                     svd_model_components[0], example_user_id, svd_model_components[1], svd_model_components[2]
                 )
                 print(f"\nSVD Recommendations:")
                 print(svd_recs)
             except Exception as e:
                 print(f"Could not generate SVD recommendations: {e}")

        if als_model_components:
            try:
                als_user_idx = train_user_map[train_user_map['user_id'] == example_user_id].index[0]
                als_recs = collaborative_filtering.get_als_recommendations(
                    als_model_components[0], als_user_idx, als_model_components[1], als_model_components[3] # model, sparse_matrix, item_map
                )
                print(f"\nALS Recommendations:")
                print(als_recs)
            except Exception as e:
                 print(f"Could not generate ALS recommendations: {e}")

        if cb_model_components:
             try:
                 user_liked_items = train_df_indexed[train_df_indexed['user_id'] == example_user_id]['item_id'].tolist()
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
        print("Skipping example recommendations (training data is empty).")


    print("\n--- Pipeline Finished ---")

if __name__ == '__main__':
    run_pipeline()