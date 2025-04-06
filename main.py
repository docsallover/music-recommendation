# main.py
"""
Main script to run the recommendation system pipeline.
"""
import pandas as pd
import data_loader
import preprocessing
import eda
import content_based
import collaborative_filtering
import evaluation
import config

def run_pipeline():
    """ Executes the main steps of the recommendation system project. """
    print("--- Starting Music Recommendation Pipeline ---")

    # 1. Load Data
    print("\n--- 1. Loading Data ---")
    df_interactions = data_loader.load_interaction_data()
    df_metadata = data_loader.load_metadata()

    if df_interactions is None:
        print("Failed to load interaction data. Exiting.")
        return
    # Metadata is optional for some models but good for EDA and CB
    if df_metadata is None:
        print("Warning: Metadata not loaded. Content-Based filtering might not work.")

    # 2. Exploratory Data Analysis (Optional but recommended)
    print("\n--- 2. Performing EDA ---")
    # eda.plot_interaction_distribution(df_interactions) # Add other EDA calls as needed

    # 3. Preprocessing & Splitting
    print("\n--- 3. Preprocessing Data ---")
    df_filtered = preprocessing.filter_sparse_data(df_interactions)

    # Get user/item mappings *before* splitting if needed consistently across models
    # Note: Interaction matrix creation might be model-specific (e.g., Surprise handles it)
    _, user_map, item_map, df_filtered_indexed = preprocessing.create_user_item_matrix(df_filtered.copy())
    all_song_ids_set = set(item_map['song_id']) # Useful for CB


    if 'timestamp' in df_filtered_indexed.columns:
        train_df, test_df = preprocessing.time_based_split(df_filtered_indexed)
    else:
        print("Warning: Timestamp not found. Cannot perform time-based split. CF/Evaluation might be inaccurate.")
        # Handle fallback (e.g., stop, or use random split with caveats)
        train_df, test_df = train_test_split(df_filtered_indexed, test_size=config.TEST_SET_SIZE, random_state=42) # Example fallback

    if train_df is None or test_df is None:
         print("Data splitting failed. Exiting.")
         return

    # 4. Train Models
    print("\n--- 4. Training Models ---")

    # Content-Based Model (if metadata available)
    cb_model_components = None
    if df_metadata is not None and 'genre' in df_metadata.columns: # Check for necessary column
        try:
            cb_similarity_matrix, cb_indices = content_based.build_content_similarity_matrix(df_metadata, text_feature_col='genre')
            cb_model_components = (cb_similarity_matrix, cb_indices, all_song_ids_set)
            print("Content-Based similarity matrix built.")
        except Exception as e:
            print(f"Error building Content-Based model: {e}")
    else:
        print("Skipping Content-Based model training (metadata or genre column missing).")

    # Collaborative Filtering Model (SVD example)
    cf_model_components = None
    rating_col_for_cf = 'interaction_count' # Choose appropriate column
    if rating_col_for_cf not in train_df.columns:
        print(f"Warning: Rating column '{rating_col_for_cf}' not in training data. CF model might need binary ratings.")
        # Add logic to create binary rating if needed
        rating_col_for_cf = 'binary_rating' # Assuming it's created or handled in train_svd_model
        if rating_col_for_cf not in train_df.columns:
             train_df['binary_rating'] = 1 # Simplistic binary creation


    try:
        cf_algo, cf_trainset = collaborative_filtering.train_svd_model(train_df, rating_col=rating_col_for_cf)
        # Pass item_map which maps internal Surprise IDs back to original song_ids
        cf_model_components = (cf_algo, cf_trainset, item_map)
        print("Collaborative Filtering (SVD) model trained.")
    except Exception as e:
        print(f"Error training Collaborative Filtering model: {e}")


    # 5. Evaluate Models
    print("\n--- 5. Evaluating Models ---")
    if cf_model_components:
        try:
            print("\nEvaluating CF Model:")
            cf_eval_results = evaluation.evaluate_model(cf_model_components, test_df, train_df, model_type='cf')
            print(f"CF Evaluation Results: {cf_eval_results}")
        except Exception as e:
            print(f"Error evaluating CF model: {e}")

    if cb_model_components:
        try:
            print("\nEvaluating CB Model:")
            # Note: CB evaluation needs train_df to get user history
            cb_eval_results = evaluation.evaluate_model(cb_model_components, test_df, train_df, model_type='cb')
            print(f"CB Evaluation Results: {cb_eval_results}")
        except Exception as e:
             print(f"Error evaluating CB model: {e}")

    # 6. Generate Example Recommendations (Optional)
    print("\n--- 6. Generating Example Recommendations ---")
    if cf_model_components and not train_df.empty:
         example_user_id = train_df['user_id'].iloc[0] # Pick a user from training data
         print(f"\nExample CF Recommendations for User: {example_user_id}")
         try:
             cf_recs = collaborative_filtering.get_cf_recommendations(cf_model_components[0], example_user_id, cf_model_components[1], cf_model_components[2])
             print(cf_recs)
         except Exception as e:
             print(f"Could not generate CF recommendations: {e}")

    if cb_model_components and not train_df.empty:
         example_user_id_cb = train_df['user_id'].iloc[0] # Pick a user
         user_liked_songs = train_df[train_df['user_id'] == example_user_id_cb]['song_id'].tolist()
         print(f"\nExample CB Recommendations for User: {example_user_id_cb} (based on {len(user_liked_songs)} liked songs)")
         if user_liked_songs:
             try:
                 cb_recs = content_based.get_content_based_recommendations(user_liked_songs, cb_model_components[0], cb_model_components[1], cb_model_components[2])
                 print(cb_recs)
             except Exception as e:
                 print(f"Could not generate CB recommendations: {e}")
         else:
             print("User has no liked songs in training data for CB.")


    print("\n--- Pipeline Finished ---")

if __name__ == '__main__':
    run_pipeline()