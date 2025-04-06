# eda.py
"""
Functions for exploring and visualizing the data.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_interaction_data, load_metadata

def plot_interaction_distribution(df, interaction_col='interaction_count'):
    """ Plots the distribution of interaction counts/ratings. """
    if interaction_col not in df.columns:
        print(f"Warning: Interaction column '{interaction_col}' not found. Plotting item frequency instead.")
        plt.figure(figsize=(10, 6))
        sns.histplot(df['song_id'].value_counts(), bins=50, kde=False)
        plt.title('Distribution of Interactions per Song')
        plt.xlabel('Number of Interactions')
        plt.ylabel('Number of Songs')
        plt.yscale('log') # Often helpful due to long tail
        plt.show()
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(df[interaction_col], bins=50, kde=True)
    plt.title(f'Distribution of {interaction_col}')
    plt.xlabel(interaction_col)
    plt.ylabel('Frequency')
    # Consider log scale if distribution is heavily skewed
    # plt.xscale('log')
    plt.show()

def plot_top_items(df, item_col='song_id', item_name_map=None, top_n=20):
    """ Plots the most popular items. """
    top_items = df[item_col].value_counts().head(top_n)

    plt.figure(figsize=(12, 8))
    # Try to map IDs to names if map provided
    if item_name_map is not None:
        labels = [f"{item_name_map.get(idx, idx)} ({idx})" for idx in top_items.index]
    else:
        labels = top_items.index

    sns.barplot(x=top_items.values, y=labels, orient='h')
    plt.title(f'Top {top_n} Most Interacted With {item_col}s')
    plt.xlabel('Number of Interactions')
    plt.ylabel(item_col.replace('_', ' ').title())
    plt.show()

def plot_user_activity(df, user_col='user_id'):
    """ Plots the distribution of interactions per user. """
    user_activity = df[user_col].value_counts()
    plt.figure(figsize=(10, 6))
    sns.histplot(user_activity, bins=50, kde=False)
    plt.title('Distribution of Interactions per User')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Number of Users')
    plt.yscale('log') # Usually necessary
    plt.show()


if __name__ == '__main__':
    df_interactions = load_interaction_data()
    df_metadata = load_metadata()

    if df_interactions is not None:
        plot_interaction_distribution(df_interactions)
        plot_user_activity(df_interactions)

        # Create a simple map from song_id to title for plotting if metadata available
        song_title_map = None
        if df_metadata is not None and 'song_id' in df_metadata.columns and 'title' in df_metadata.columns:
            song_title_map = df_metadata.set_index('song_id')['title'].to_dict()

        plot_top_items(df_interactions, item_name_map=song_title_map)