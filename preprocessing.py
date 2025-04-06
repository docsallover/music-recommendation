# preprocessing.py
"""
Functions for data cleaning, feature engineering, and splitting.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # For basic splitting if time-based isn't feasible initially
from scipy.sparse import csr_matrix
import config

def filter_sparse_data(df, min_user_interactions=config.MIN_INTERACTIONS_PER_USER, min_item_interactions=config.MIN_INTERACTIONS_PER_ITEM):
    """ Filters out users and items with too few interactions. """
    print("Filtering sparse data...")
    # User counts
    user_counts = df['user_id'].value_counts()
    df = df[df['user_id'].isin(user_counts[user_counts >= min_user_interactions].index)]
    # Item counts
    item_counts = df['song_id'].value_counts()
    df = df[df['song_id'].isin(item_counts[item_counts >= min_item_interactions].index)]
    print(f"Data shape after filtering: {df.shape}")
    return df

def create_user_item_matrix(df, interaction_col='interaction_count'):
    """ Creates a sparse user-item interaction matrix. """
    print("Creating user-item matrix...")
    # Create contiguous IDs for users and items
    df['user_idx'] = df['user_id'].astype('category').cat.codes
    df['item_idx'] = df['song_id'].astype('category').cat.codes

    # Get mappings for later use
    user_map = df[['user_idx', 'user_id']].drop_duplicates().set_index('user_idx')
    item_map = df[['item_idx', 'song_id']].drop_duplicates().set_index('item_idx')

    # Create the sparse matrix
    # Use interaction_col if available, otherwise assume binary interaction (1)
    interaction_values = df[interaction_col] if interaction_col in df.columns else np.ones(df.shape[0])
    sparse_matrix = csr_matrix((interaction_values, (df['user_idx'], df['item_idx'])),
                              shape=(df['user_idx'].nunique(), df['item_idx'].nunique()))

    print(f"Created sparse matrix with shape: {sparse_matrix.shape}")
    return sparse_matrix, user_map, item_map, df # Return df with new indices

def time_based_split(df, test_size=config.TEST_SET_SIZE):
    """
    Splits data into train and test sets based on timestamp.
    Assumes df is sorted by timestamp ASCENDING.
    """
    print("Performing time-based split...")
    if 'timestamp' not in df.columns:
        raise ValueError("Timestamp column required for time-based split.")

    # Ensure data is sorted by time
    df = df.sort_values('timestamp').reset_index(drop=True)

    n_test = int(len(df) * test_size)
    n_train = len(df) - n_test

    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]

    # Ensure test set doesn't contain users/items not in train set (can be handled differently)
    test_df = test_df[test_df['user_id'].isin(train_df['user_id'])]
    test_df = test_df[test_df['song_id'].isin(train_df['song_id'])]

    print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
    return train_df, test_df


if __name__ == '__main__':
    from data_loader import load_interaction_data
    df_interactions = load_interaction_data()

    if df_interactions is not None:
        df_filtered = filter_sparse_data(df_interactions)

        if 'timestamp' in df_filtered.columns:
             train_data, test_data = time_based_split(df_filtered)
        else:
             print("Warning: Timestamp column not found. Cannot perform time-based split.")
             # Fallback or stop execution
             train_data, test_data = None, None # Or implement random split as fallback

        if train_data is not None:
            # Example: Create matrix only from training data for model training
            sparse_matrix, user_map, item_map, train_data_indexed = create_user_item_matrix(train_data)
            print("\nUser Map Head:")
            print(user_map.head())
            print("\nItem Map Head:")
            print(item_map.head())