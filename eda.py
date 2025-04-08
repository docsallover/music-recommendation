# eda.py
"""
Functions for exploring and visualizing the data.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import (
    load_lastfm_data,
)  # Assuming initial load here for standalone run
from preprocessing import create_item_id  # Needed to get item_id


def plot_interaction_distribution(df, interaction_col="interaction_count"):
    """Plots the distribution of interaction counts/ratings."""
    if interaction_col not in df.columns:
        print(
            f"Warning: Interaction column '{interaction_col}' not found. Plotting item frequency instead."
        )
        plt.figure(figsize=(10, 6))
        sns.histplot(df["song_id"].value_counts(), bins=50, kde=False)
        plt.title("Distribution of Interactions per Song")
        plt.xlabel("Number of Interactions")
        plt.ylabel("Number of Songs")
        plt.yscale("log")  # Often helpful due to long tail
        plt.show()
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(df[interaction_col], bins=50, kde=True)
    plt.title(f"Distribution of {interaction_col}")
    plt.xlabel(interaction_col)
    plt.ylabel("Frequency")
    # Consider log scale if distribution is heavily skewed
    # plt.xscale('log')
    plt.show()


def plot_top_items(df, item_col="song_id", item_name_map=None, top_n=20):
    """Plots the most popular items."""
    top_items = df[item_col].value_counts().head(top_n)

    plt.figure(figsize=(12, 8))
    # Try to map IDs to names if map provided
    if item_name_map is not None:
        labels = [f"{item_name_map.get(idx, idx)} ({idx})" for idx in top_items.index]
    else:
        labels = top_items.index

    sns.barplot(x=top_items.values, y=labels, orient="h")
    plt.title(f"Top {top_n} Most Interacted With {item_col}s")
    plt.xlabel("Number of Interactions")
    plt.ylabel(item_col.replace("_", " ").title())
    plt.show()


def plot_user_activity(df, user_col="user_id"):
    """Plots the distribution of interactions per user."""
    user_activity = df[user_col].value_counts()
    plt.figure(figsize=(10, 6))
    sns.histplot(user_activity, bins=50, kde=False)
    plt.title("Distribution of Interactions per User")
    plt.xlabel("Number of Interactions")
    plt.ylabel("Number of Users")
    plt.yscale("log")  # Usually necessary
    plt.show()


# eda.py
"""
Functions for exploring and visualizing the Last.fm data.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import (
    load_lastfm_data,
)  # Assuming initial load here for standalone run
from preprocessing import create_item_id  # Needed to get item_id

# Keep plot_interaction_distribution, plot_user_activity as before, they use user_id/item_id count


def plot_top_artists(df, artist_col="artist", top_n=20):
    """Plots the most popular artists based on listen events."""
    if artist_col not in df.columns:
        print(f"Column '{artist_col}' not found.")
        return
    top_items = df[artist_col].value_counts().head(top_n)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_items.values, y=top_items.index, orient="h")
    plt.title(f"Top {top_n} Most Listened To Artists")
    plt.xlabel("Number of Listen Events")
    plt.ylabel("Artist")
    plt.tight_layout()
    plt.show()


def plot_top_tracks(df, item_col="item_id", top_n=20):
    """Plots the most popular tracks (Artist - Track)."""
    if item_col not in df.columns:
        print(f"Column '{item_col}' not found. Did you run create_item_id first?")
        return
    top_items = df[item_col].value_counts().head(top_n)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_items.values, y=top_items.index, orient="h")
    plt.title(f"Top {top_n} Most Listened To Tracks (Artist - Track)")
    plt.xlabel("Number of Listen Events")
    plt.ylabel("Track (Artist - Track)")
    plt.tight_layout()
    plt.show()


def plot_listens_over_time(df, time_col="timestamp", freq="D"):
    """Plots the number of listen events over time."""
    if time_col not in df.columns:
        print(f"Column '{time_col}' not found.")
        return
    listens_ts = df.set_index(time_col).resample(freq).size()
    plt.figure(figsize=(15, 5))
    listens_ts.plot()
    plt.title(f"Listen Events Over Time ({freq} Frequency)")
    plt.xlabel("Date")
    plt.ylabel("Number of Listens")
    plt.show()


if __name__ == "__main__":
    df_interactions = load_lastfm_data()

    if df_interactions is not None:
        df_with_itemid = create_item_id(df_interactions)  # Create item_id for plotting

        # --- Call EDA functions ---
        print("\n--- Running EDA ---")
        plot_user_activity(df_with_itemid, user_col="user_id")
        plot_top_artists(df_with_itemid)
        plot_top_tracks(df_with_itemid)  # Uses item_id
        plot_listens_over_time(df_with_itemid)
