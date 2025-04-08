# content_based.py
"""
Implementation of Content-Based Filtering using Artist/Album/Track text features.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config

# No separate metadata load needed, get info from main interaction df


def get_item_metadata_df(
    df_interactions,
    item_col="item_id",
    artist_col="artist",
    album_col="album",
    track_col="track",
):
    """Extracts unique item metadata from the interaction dataframe."""
    if item_col not in df_interactions.columns:
        raise ValueError(f"Item ID column '{item_col}' not found in interactions.")

    metadata_cols = [item_col, artist_col, album_col, track_col]
    if not all(col in df_interactions.columns for col in metadata_cols):
        print(
            f"Warning: Missing some metadata columns ({metadata_cols}). Proceeding with available ones."
        )
        metadata_cols = [col for col in metadata_cols if col in df_interactions.columns]

    df_metadata = (
        df_interactions[metadata_cols]
        .drop_duplicates(subset=[item_col])
        .reset_index(drop=True)
    )
    print(f"Extracted unique item metadata for {df_metadata.shape[0]} items.")
    return df_metadata


def build_content_similarity_matrix(
    df_item_metadata, item_col="item_id", feature_cols=["artist", "album", "track"]
):
    """
    Builds an item-item similarity matrix based on text features from specified columns.

    Args:
        df_item_metadata (pd.DataFrame): DataFrame with unique items and their metadata.
        item_col (str): Column name for the unique item ID.
        feature_cols (list): List of column names containing text features to combine.

    Returns:
        tuple: (similarity_matrix, pd.Series mapping matrix index to item_id)
    """
    print(f"Building content similarity matrix based on features: {feature_cols}...")

    # Check if required columns exist
    if item_col not in df_item_metadata.columns:
        raise ValueError(f"Metadata missing required column: '{item_col}'")
    valid_feature_cols = [
        col for col in feature_cols if col in df_item_metadata.columns
    ]
    if not valid_feature_cols:
        raise ValueError(
            "No valid feature columns found in metadata for similarity calculation."
        )
    print(f"Using valid feature columns: {valid_feature_cols}")

    # Combine text features into a single string per item, handling potential NaN
    def combine_features(row):
        combined = []
        for col in valid_feature_cols:
            # Convert to string and handle NaN explicitly
            feature_val = str(row[col]) if pd.notna(row[col]) else ""
            combined.append(feature_val)
        return " ".join(combined).lower()  # Combine and lowercas

    df_item_metadata["combined_features"] = df_item_metadata.apply(
        combine_features, axis=1
    )

    # Use TF-IDF to vectorize the combined features
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df_item_metadata["combined_features"])

    # Calculate cosine similarity between items
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Create mapping from matrix index to item_id
    indices = pd.Series(df_item_metadata.index, index=df_item_metadata[item_col])

    print(f"Content similarity matrix shape: {cosine_sim.shape}")
    return cosine_sim, indices


# get_content_based_recommendations function remains largely the same,
# just ensure it uses the correct 'item_id' format and receives the
# correct 'all_song_ids' set (which should be all unique 'item_id's from preprocessing).
# Renaming 'song_id' to 'item_id' in args/docs for clarity.


def get_content_based_recommendations(
    user_liked_items,
    cosine_sim,
    item_indices,
    all_item_ids,
    top_n=config.N_RECOMMENDATIONS,
):
    """
    Generates recommendations for a user based on content similarity.

    Args:
        user_liked_items (list): List of item_ids the user has interacted positively with.
        cosine_sim (np.ndarray): Precomputed item-item cosine similarity matrix.
        item_indices (pd.Series): Maps item_id to matrix index.
        all_item_ids (set): Set of all valid item IDs in the system.
        top_n (int): Number of recommendations to return.

    Returns:
        list: List of recommended item_ids.
    """
    # Filter out items not in our metadata/similarity matrix
    valid_liked_items = [item for item in user_liked_items if item in item_indices]
    if not valid_liked_items:
        # print("Warning: User has no liked items present in the metadata index.")
        return []  # Or return popular items as fallback

    # Get indices of liked items
    try:
        liked_indices = item_indices[valid_liked_items].tolist()
    except KeyError as e:
        print(
            f"Warning: Some liked items not found in item_indices: {e}. Skipping them."
        )
        valid_liked_items = [item for item in valid_liked_items if item in item_indices]
        if not valid_liked_items:
            return []
        liked_indices = item_indices[valid_liked_items].tolist()

    # Aggregate similarity scores from all liked items
    avg_sim_scores = cosine_sim[liked_indices].mean(axis=0)

    # Convert scores to a Series with item_ids as index
    sim_scores_series = pd.Series(avg_sim_scores, index=item_indices.index)

    # Sort by similarity score
    sim_scores_series = sim_scores_series.sort_values(ascending=False)

    # Filter out already liked items and get top N
    recommended_items = []
    for item_id, score in sim_scores_series.items():
        # Ensure item exists in the overall system (all_item_ids from interaction data)
        if item_id not in user_liked_items and item_id in all_item_ids:
            recommended_items.append(item_id)
        if len(recommended_items) >= top_n:
            break

    return recommended_items


if __name__ == "__main__":
    from data_loader import load_lastfm_data
    from preprocessing import (
        create_item_id,
        filter_sparse_data,
    )  # Need preprocessing steps

    df_inter = load_lastfm_data()
    if df_inter is not None:
        df_with_id = create_item_id(df_inter)
        df_filtered = filter_sparse_data(
            df_with_id
        )  # Filter before extracting metadata

        # Extract unique metadata from the filtered interaction data
        df_meta = get_item_metadata_df(df_filtered)

        if not df_meta.empty:
            # Define features to use for similarity
            feature_list = ["artist", "album", "track"]
            try:
                similarity_matrix, item_indices_map = build_content_similarity_matrix(
                    df_meta, feature_cols=feature_list
                )

                # Example usage:
                # Get a user and their liked items from the filtered data
                example_user = df_filtered["user_id"].iloc[0]
                user_history = (
                    df_filtered[df_filtered["user_id"] == example_user]["item_id"]
                    .unique()
                    .tolist()
                )

                all_items_in_system = set(
                    df_filtered["item_id"]
                )  # All items after filtering

                if user_history:
                    recommendations = get_content_based_recommendations(
                        user_liked_items=user_history,
                        cosine_sim=similarity_matrix,
                        item_indices=item_indices_map,
                        all_item_ids=all_items_in_system,
                    )
                    print(
                        f"\nContent-Based Recommendations for user '{example_user}' (history size {len(user_history)}):"
                    )
                    # Print first few recommendations
                    print(recommendations[: config.N_RECOMMENDATIONS])
                else:
                    print(
                        f"\nUser '{example_user}' has no history in the filtered data."
                    )

            except ValueError as e:
                print(f"\nCould not run Content-Based example: {e}")
        else:
            print("\nCould not extract metadata for Content-Based example.")
