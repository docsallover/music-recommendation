# main.py
"""
Main script to run the Last.fm recommendation system pipeline.
Implements Content-Based and SVD Collaborative Filtering.
Prompts user for Username for example recommendations.
"""
import pandas as pd
import data_loader
import preprocessing
import eda
import content_based
import collaborative_filtering  # Only imports SVD functions now
import evaluation
import config


def run_pipeline():
    """Executes the main steps of the recommendation system project."""
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

    print("\n--- Splitting Data (Time-Based) ---")
    train_df, test_df = preprocessing.time_based_split(df_filtered)

    # --- Prepare components needed for different models ---
    print("\n--- Preparing Training Set Components ---")
    train_item_metadata = content_based.get_item_metadata_df(train_df)
    all_items_in_train = set(train_item_metadata["item_id"])

    # Create sparse matrix and mappings from training data (needed for SVD item map)
    # Use train_df.copy() to avoid modifying the original split dataframe
    # Note: SVD itself doesn't need the matrix, but we need the item_map from this process
    # Store the user_map as well for validation later
    _, train_user_map, train_item_map, _ = preprocessing.create_user_item_matrix(
        train_df.copy()
    )

    # 5. Train Models
    print("\n--- 5. Training Models ---")

    # Content-Based Model
    cb_model_components = None
    if not train_item_metadata.empty:
        try:
            print("Training Content-Based Model...")
            cb_feature_cols = ["artist", "album", "track"]  # Choose features
            cb_similarity_matrix, cb_indices = (
                content_based.build_content_similarity_matrix(
                    train_item_metadata, feature_cols=cb_feature_cols
                )
            )
            cb_model_components = (cb_similarity_matrix, cb_indices, all_items_in_train)
            print("Content-Based similarity matrix built.")
        except Exception as e:
            print(f"Error building Content-Based model: {e}")
    else:
        print(
            "Skipping Content-Based model training (no metadata extracted from train set)."
        )

    # Collaborative Filtering: SVD (Surprise)
    svd_model_components = None
    try:
        print("Training CF Model (SVD)...")
        svd_algo, svd_trainset = collaborative_filtering.train_svd_model_implicit(
            train_df
        )
        svd_model_components = (
            svd_algo,
            svd_trainset,
            train_item_map,
        )  # Use item_map derived from train_df
        print("SVD model trained.")
    except Exception as e:
        print(f"Error training SVD model: {e}")

    # 6. Evaluate Models (only if test_df is not empty)
    print("\n--- 6. Evaluating Models ---")
    if not test_df.empty:
        if svd_model_components:
            evaluation.evaluate_model(
                svd_model_components, test_df, train_df, model_type="svd"
            )
        else:
            print("Skipping SVD evaluation (model not trained).")

        if cb_model_components:
            evaluation.evaluate_model(
                cb_model_components, test_df, train_df, model_type="cb"
            )
        else:
            print("Skipping Content-Based evaluation (model not trained).")
    else:
        print("Skipping evaluation because the test set is empty.")

    # 7. Generate Example Recommendations (Interactive)
    print("\n--- 7. Generating Example Recommendations ---")
    # Ensure train data and maps exist to proceed
    # Use train_user_map which contains the standardized 'user_id' column (holding original usernames)
    if (
        not train_df.empty
        and not train_item_map.empty
        and not train_user_map.empty
        and "user_id" in train_user_map.columns
    ):

        # --- Get User Input ---
        # Get valid original usernames from the 'user_id' column in train_user_map
        valid_usernames = set(train_user_map["user_id"])

        if not valid_usernames:
            print(
                "No valid users found in the training map to generate recommendations for."
            )
        else:
            # Prompt user for input
            try:
                # <<< MODIFIED PROMPT >>>
                entered_username = input(
                    f"Enter a Username to generate recommendations for (e.g., one of the {len(valid_usernames)} users in train set): "
                ).strip()
            except EOFError:
                print("\nNo input received. Skipping example recommendations.")
                entered_username = None  # Ensure variable exists

            if entered_username is not None:
                # Validate user input (check if entered username is in the set of valid usernames)
                if entered_username in valid_usernames:
                    print(
                        f"\n--- Generating Recommendations for Username: {entered_username} ---"
                    )  # <<< MODIFIED PRINT >>>

                    # --- SVD Recommendations ---
                    # Pass the entered_username (which corresponds to the original Username value)
                    if svd_model_components:
                        try:
                            svd_recs = (
                                collaborative_filtering.get_cf_recommendations_surprise(
                                    svd_model_components[0],
                                    entered_username,
                                    svd_model_components[1],
                                    svd_model_components[2],  # algo, trainset, item_map
                                )
                            )
                            print(f"\nSVD Recommendations:")
                            if svd_recs:
                                print(svd_recs)
                            else:
                                print(
                                    "(No recommendations generated by SVD model - user might be unknown to internal model or have interacted with many items)"
                                )
                        except Exception as e:
                            print(f"Could not generate SVD recommendations: {e}")
                    else:
                        print("\nSkipping SVD recommendations (model not trained).")

                    # --- Content-Based Recommendations ---
                    # Use entered_username to filter train_df
                    if cb_model_components:
                        try:
                            # Get user history from the original train_df
                            user_liked_items = (
                                train_df[train_df["user_id"] == entered_username][
                                    "item_id"
                                ]
                                .unique()
                                .tolist()
                            )
                            if user_liked_items:
                                cb_recs = (
                                    content_based.get_content_based_recommendations(
                                        user_liked_items,
                                        cb_model_components[0],
                                        cb_model_components[1],
                                        cb_model_components[
                                            2
                                        ],  # sim_matrix, indices, all_items
                                    )
                                )
                                print(
                                    f"\nContent-Based Recommendations (based on {len(user_liked_items)} items):"
                                )
                                if cb_recs:
                                    print(cb_recs)
                                else:
                                    print("(No recommendations generated by CB model)")
                            else:
                                print(
                                    "\nUser has no known history in training data for Content-Based recommendations."
                                )
                        except Exception as e:
                            print(f"Could not generate CB recommendations: {e}")
                    else:
                        print("\nSkipping CB recommendations (model not trained).")

                else:
                    # <<< MODIFIED ERROR MESSAGE >>>
                    print(
                        f"Error: Username '{entered_username}' not found in the training set users."
                    )
                    # Optional: Print some example valid usernames to help the user
                    # print("Example valid usernames:", list(valid_usernames)[:5])

    else:
        print(
            "Skipping example recommendations (training data or necessary maps are empty)."
        )

    print("\n--- Pipeline Finished ---")


if __name__ == "__main__":
    run_pipeline()
