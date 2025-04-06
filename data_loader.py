# data_loader.py
"""
Functions to load and perform initial processing on the Last.fm data.
"""
import pandas as pd
import config
import os

def load_lastfm_data(file_path=config.LASTFM_DATA_PATH):
    """
    Loads the Last.fm interaction data and performs initial cleaning.

    Args:
        file_path (str): Path to the Last.fm data file.

    Returns:
        pandas.DataFrame: DataFrame with interaction data, standardized column names,
                          and combined timestamp. Returns None on failure.
                          Columns: 'user_id', 'artist', 'track', 'album', 'timestamp'
    """
    if not os.path.exists(file_path):
        print(f"Error: Last.fm data file not found at {file_path}")
        print("Please update the LASTFM_DATA_PATH in config.py")
        return None

    try:
        # TODO: Adjust read_csv parameters if your file has different separator, encoding etc.
        df = pd.read_csv(file_path)
        print(f"Loaded Last.fm data: {df.shape[0]} rows, {df.shape[1]} columns")

        # Define expected columns based on description
        expected_cols = ['Username', 'Artist', 'Track', 'Album', 'Date', 'Time']
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"Data missing expected columns. Found: {df.columns.tolist()}")

        # --- Data Cleaning & Standardization ---
        # 1. Rename columns for consistency
        df = df.rename(columns={
            'Username': 'user_id',
            'Artist': 'artist',
            'Track': 'track',
            'Album': 'album',
            'Date': 'date_str',
            'Time': 'time_str'
        })

        # 2. Handle missing values (especially for columns used to create item_id)
        #    Decide strategy: drop rows, fill with placeholder like 'Unknown'
        original_rows = len(df)
        df.dropna(subset=['user_id', 'artist', 'track'], inplace=True)
        if len(df) < original_rows:
            print(f"Dropped {original_rows - len(df)} rows with missing user_id, artist, or track.")
        # Fill missing album names if needed
        df['album'] = df['album'].fillna('Unknown Album')


        # 3. Combine Date and Time into a timestamp
        #    Try common formats, adjust format string if needed
        try:
            df['timestamp'] = pd.to_datetime(df['date_str'] + ' ' + df['time_str'], errors='coerce') # errors='coerce' will turn unparseable dates into NaT
        except Exception as e:
             print(f"Could not combine Date and Time automatically: {e}. Trying specific format...")
             # *** ADJUST THE format string below if your date/time format is different ***
             # Example: If Date='31-Jan-2021' and Time='23:59', use format='%d-%b-%Y %H:%M'
             try:
                 df['timestamp'] = pd.to_datetime(df['date_str'] + ' ' + df['time_str'], format='%d-%b-%Y %H:%M', errors='coerce') # Example Format
             except Exception as e2:
                  print(f"Specific format failed too: {e2}. Timestamp creation failed.")
                  return None # Cannot proceed without timestamp for time-split

        df.dropna(subset=['timestamp'], inplace=True) # Drop rows where timestamp parsing failed
        if len(df) < original_rows:
             print(f"Dropped {original_rows - len(df)} rows due to missing or unparseable date/time.")

        # 4. Select and order columns
        final_cols = ['user_id', 'artist', 'track', 'album', 'timestamp']
        df = df[final_cols]

        print(f"Data loaded and initially processed: {df.shape[0]} rows")
        return df

    except FileNotFoundError:
        # This is redundant due to the check at the start, but good practice
        print(f"Error: Last.fm data file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading or processing Last.fm data: {e}")
        return None

# Remove the separate load_metadata function as metadata is in the main file now.

if __name__ == '__main__':
    # Example usage:
    interactions_df = load_lastfm_data()
    if interactions_df is not None:
        print("\nProcessed Last.fm Data Head:")
        print(interactions_df.head())
        print("\nData Info:")
        interactions_df.info()