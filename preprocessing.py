# preprocessing.py
"""
Functions for data cleaning specific to interactions, feature engineering, and splitting.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Fallback split
from scipy.sparse import csr_matrix
import config

def create_item_id(df, artist_col='artist', track_col='track', sep=config.ITEM_ID_SEPARATOR):
    """ Creates a unique item ID by combining artist and track. """
    print("Creating unique 'item_id' from artist and track...")
    df['item_id'] = df[artist_col].astype(str) + sep + df[track_col].astype(str)
    return df

def filter_sparse_data(df, user_col='user_id', item_col='item_id',
                       min_user_interactions=config.MIN_INTERACTIONS_PER_USER,
                       min_item_interactions=config.MIN_INTERACTIONS_PER_ITEM):
    """ Filters out users and items with too few interactions. """
    print(f"Filtering sparse data (min user: {min_user_interactions}, min item: {min_item_interactions})...")
    original_shape = df.shape
    while True:
        # User counts
        user_counts = df.groupby(user_col).size()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df_filtered_users = df[df[user_col].isin(valid_users)]

        # Item counts
        item_counts = df_filtered_users.groupby(item_col).size()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df_filtered_items = df_filtered_users[df_filtered_users[item_col].isin(valid_items)]

        # Check if filtering changed the dataframe size
        if df_filtered_items.shape == df.shape:
            break # No more filtering needed
        else:
            df = df_filtered_items # Continue filtering

    print(f"Data shape changed from {original_shape} to {df.shape} after filtering.")
    # Re-check user counts after item filtering
    final_user_counts = df.groupby(user_col).size()
    final_valid_users = final_user_counts[final_user_counts >= min_user_interactions].index
    df = df[df[user_col].isin(final_valid_users)]
    print(f"Final data shape after re-checking user counts: {df.shape}")
    return df

def create_user_item_matrix(df, user_col='user_id', item_col='item_id'):
    """
    Creates a sparse user-item interaction matrix (binary interaction).
    Assumes df contains one row per listen event.
    """
    print("Creating user-item matrix (binary interactions)...")
    print(f"DEBUG: Input df shape: {df.shape}")
    print(f"DEBUG: Input unique users: {df[user_col].nunique()}")

    # Important: Drop duplicate user-item interactions if treating as binary (1 listen = interaction)
    df = df.sort_values('timestamp').drop_duplicates(subset=[user_col, item_col], keep='first')
    print(f"DEBUG: Shape after dropping duplicates: {df.shape}")
    print(f"DEBUG: Unique users after dropping duplicates: {df[user_col].nunique()}") # <<< ADD THIS

    if df.empty:
        print("ERROR: DataFrame is empty after dropping duplicates!")
        # Return empty structures or raise an error
        empty_matrix = csr_matrix((0, 0))
        empty_map = pd.DataFrame()
        return empty_matrix, empty_map, empty_map, df

    # Create contiguous IDs for users and items
    df['user_idx'] = df[user_col].astype('category').cat.codes
    df['item_idx'] = df[item_col].astype('category').cat.codes
    print(f"DEBUG: Unique user_idx count after .cat.codes: {df['user_idx'].nunique()}") # <<< ADD THIS
    print(f"DEBUG: Max user_idx: {df['user_idx'].max()}") # <<< ADD THIS
    print(f"DEBUG: Unique item_idx count after .cat.codes: {df['item_idx'].nunique()}") # <<< ADD THIS

    # Get mappings for later use
    user_map = df[['user_idx', user_col]].drop_duplicates().set_index('user_idx')
    item_map = df[['item_idx', item_col]].drop_duplicates().set_index('item_idx')
    item_info = df[['item_idx', 'artist', 'track']].drop_duplicates().set_index('item_idx')
    item_map = item_map.join(item_info)

    # Create the sparse matrix with binary interaction values (1)
    interaction_values = np.ones(df.shape[0], dtype=np.float32)
    num_users = df['user_idx'].nunique() # Use calculated nunique value
    num_items = df['item_idx'].nunique() # Use calculated nunique value

    # <<< ADDED Check >>>
    if num_users <= 0 or num_items <= 0:
         print(f"ERROR: Invalid matrix dimensions calculated: ({num_users}, {num_items})")
         empty_matrix = csr_matrix((0, 0))
         empty_map = pd.DataFrame()
         return empty_matrix, empty_map, empty_map, df


    print(f"DEBUG: Creating matrix with explicit shape: ({num_users}, {num_items})") # <<< ADD THIS
    sparse_matrix = csr_matrix((interaction_values, (df['user_idx'], df['item_idx'])),
                              shape=(num_users, num_items)) # Use calculated shape

    print(f"Created sparse matrix with shape: {sparse_matrix.shape}")
    return sparse_matrix, user_map, item_map, df


def time_based_split(df, test_size=config.TEST_SET_SIZE, time_col='timestamp'):
    """
    Splits data into train and test sets based on timestamp.
    Ensures test users/items exist in train set.
    """
    print(f"Performing time-based split (test size: {test_size})...")
    if time_col not in df.columns:
        raise ValueError(f"Timestamp column '{time_col}' required for time-based split.")

    # Ensure data is sorted by time
    df = df.sort_values(time_col).reset_index(drop=True)

    split_index = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    # Filter test set to include only users and items present in the train set
    # This is crucial for many evaluation methods and CF models
    train_users = set(train_df['user_id'])
    train_items = set(train_df['item_id'])
    test_df = test_df[test_df['user_id'].isin(train_users) & test_df['item_id'].isin(train_items)]

    print(f"Train set size: {len(train_df)}, Test set size (filtered): {len(test_df)}")
    return train_df, test_df


if __name__ == '__main__':
    from data_loader import load_lastfm_data
    df_interactions = load_lastfm_data()

    if df_interactions is not None:
        df_with_itemid = create_item_id(df_interactions)
        df_filtered = filter_sparse_data(df_with_itemid)

        train_data, test_data = time_based_split(df_filtered)

        if train_data is not None and not train_data.empty:
             # Create matrix only from training data for model training
             sparse_matrix, user_map, item_map, train_data_indexed = create_user_item_matrix(train_data.copy()) # Use copy
             print("\nUser Map Head:")
             print(user_map.head())
             print("\nItem Map Head:")
             print(item_map.head())
        else:
             print("Training data is empty after split/filtering.")