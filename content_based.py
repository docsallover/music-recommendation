# content_based.py
"""
Implementation of Content-Based Filtering.
Focuses on using item metadata (e.g., genre).
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import config
from data_loader import load_metadata

def build_content_similarity_matrix(df_metadata, text_feature_col='genre'):
    """
    Builds an item-item similarity matrix based on text features.

    Args:
        df_metadata (pd.DataFrame): DataFrame with song metadata including 'song_id' and text_feature_col.
        text_feature_col (str): The column containing text features (e.g., genre, artist).

    Returns:
        tuple: (similarity_matrix, pd.Series mapping matrix index to song_id)
    """
    print(f"Building content similarity matrix based on '{text_feature_col}'...")
    if text_feature_col not in df_metadata.columns:
        raise ValueError(f"Metadata missing required column: '{text_feature_col}'")

    # Ensure song_id is present and handle potential duplicates (keep first)
    df_metadata = df_metadata.drop_duplicates(subset=['song_id']).reset_index(drop=True)
    # Handle missing text features
    df_metadata[text_feature_col] = df_metadata[text_feature_col].fillna('')

    # Use TF-IDF to vectorize the text feature
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_metadata[text_feature_col])

    # Calculate cosine similarity between items
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Create mapping from matrix index to song_id
    indices = pd.Series(df_metadata.index, index=df_metadata['song_id'])

    print(f"Similarity matrix shape: {cosine_sim.shape}")
    return cosine_sim, indices

def get_content_based_recommendations(user_liked_songs, cosine_sim, item_indices, all_song_ids, top_n=config.N_RECOMMENDATIONS):
    """
    Generates recommendations for a user based on content similarity.

    Args:
        user_liked_songs (list): List of song_ids the user has interacted positively with.
        cosine_sim (np.ndarray): Precomputed item-item cosine similarity matrix.
        item_indices (pd.Series): Maps song_id to matrix index.
        all_song_ids (set): Set of all valid song IDs in the system.
        top_n (int): Number of recommendations to return.

    Returns:
        list: List of recommended song_ids.
    """
    # Filter out songs not in our metadata/similarity matrix
    valid_liked_songs = [song for song in user_liked_songs if song in item_indices]
    if not valid_liked_songs:
        print("Warning: User has no liked songs present in the metadata index.")
        return [] # Or return popular items as fallback

    # Get indices of liked songs
    liked_indices = item_indices[valid_liked_songs].tolist()

    # Calculate average similarity score for all items based on liked items
    # Aggregate similarity scores from all liked items
    avg_sim_scores = cosine_sim[liked_indices].mean(axis=0)


    # Convert scores to a Series with song_ids as index
    sim_scores_series = pd.Series(avg_sim_scores, index=item_indices.index)

    # Sort by similarity score
    sim_scores_series = sim_scores_series.sort_values(ascending=False)

    # Filter out already liked songs and get top N
    recommended_songs = []
    for song_id, score in sim_scores_series.items():
        if song_id not in user_liked_songs and song_id in all_song_ids:
             recommended_songs.append(song_id)
        if len(recommended_songs) >= top_n:
            break

    return recommended_songs


if __name__ == '__main__':
    df_meta = load_metadata()
    if df_meta is not None and 'genre' in df_meta.columns: # Make sure genre exists
        # TODO: Choose appropriate text feature column
        similarity_matrix, song_indices = build_content_similarity_matrix(df_meta, text_feature_col='genre')

        # Example usage:
        # Assume we know user 'user123' liked ['songA', 'songB']
        # And we have the full set of song IDs from the interaction data preprocessing step
        example_user_likes = ['songA', 'songB'] # Replace with actual IDs from your data
        all_songs = set(df_meta['song_id']) # Get all unique song IDs

        # Filter to ensure example likes exist in the index
        example_user_likes_filtered = [s for s in example_user_likes if s in song_indices]

        if example_user_likes_filtered:
             recommendations = get_content_based_recommendations(
                 user_liked_songs=example_user_likes_filtered,
                 cosine_sim=similarity_matrix,
                 item_indices=song_indices,
                 all_song_ids=all_songs
             )
             print(f"\nContent-Based Recommendations for user liking {example_user_likes_filtered}:")
             print(recommendations)
        else:
            print("\nCould not generate recommendations: Example liked songs not found in metadata index.")

    else:
        print("\nCould not run Content-Based example: Metadata or 'genre' column missing.")