# data_loader.py
"""
Functions to load interaction and metadata files.
"""
import pandas as pd
import config

def load_interaction_data(file_path=config.INTERACTION_DATA_PATH):
    """
    Loads user-item interaction data.

    Args:
        file_path (str): Path to the interaction data file.

    Returns:
        pandas.DataFrame: DataFrame with interaction data.
                          Expected columns: 'user_id', 'song_id', 'interaction_count/rating', 'timestamp'
    """
    try:
        # TODO: Adjust read_csv parameters based on your dataset format (sep, header, column names)
        df_interactions = pd.read_csv(file_path)
        print(f"Loaded interaction data: {df_interactions.shape[0]} rows")
        # Basic validation
        required_cols = ['user_id', 'song_id'] # Add 'interaction_count' or 'rating', 'timestamp' if needed
        if not all(col in df_interactions.columns for col in required_cols):
             raise ValueError(f"Interaction data missing required columns: {required_cols}")
        # Convert timestamp if present
        if 'timestamp' in df_interactions.columns:
            df_interactions['timestamp'] = pd.to_datetime(df_interactions['timestamp'])
        return df_interactions
    except FileNotFoundError:
        print(f"Error: Interaction data file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading interaction data: {e}")
        return None


def load_metadata(file_path=config.SONG_METADATA_PATH):
    """
    Loads song metadata.

    Args:
        file_path (str): Path to the metadata file.

    Returns:
        pandas.DataFrame: DataFrame with song metadata.
                          Expected columns: 'song_id', 'title', 'artist_name', 'genre', etc.
    """
    try:
        # TODO: Adjust read_csv parameters based on your dataset format
        df_metadata = pd.read_csv(file_path)
        print(f"Loaded metadata: {df_metadata.shape[0]} rows")
         # Basic validation
        if 'song_id' not in df_metadata.columns:
             raise ValueError("Metadata missing required column: 'song_id'")
        return df_metadata
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    interactions = load_interaction_data()
    if interactions is not None:
        print("\nInteraction Data Head:")
        print(interactions.head())

    metadata = load_metadata()
    if metadata is not None:
        print("\nMetadata Head:")
        print(metadata.head())