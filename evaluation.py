# evaluation.py
"""
Functions to evaluate recommendation models using ranking metrics.
"""
import numpy as np
import config

def precision_recall_at_k(predictions, true_relevant_items, k=config.N_RECOMMENDATIONS):
    """
    Calculates Precision@k and Recall@k for a single user.

    Args:
        predictions (list): List of recommended item_ids.
        true_relevant_items (set): Set of item_ids the user actually liked/interacted with in the test set.
        k (int): Number of recommendations to consider.

    Returns:
        tuple: (precision@k, recall@k)
    """
    predictions_at_k = predictions[:k]
    relevant_in_predictions = set(predictions_at_k) & true_relevant_items
    num_relevant_in_predictions = len(relevant_in_predictions)

    precision = num_relevant_in_predictions / k if k > 0 else 0
    recall = num_relevant_in_predictions / len(true_relevant_items) if len(true_relevant_items) > 0 else 0

    return precision, recall

def evaluate_model(model, test_data, train_data, user_col='user_id', item_col='song_id', k=config.N_RECOMMENDATIONS, model_type='cf'):
    """
    Evaluates a recommendation model over a test set.

    Args:
        model: The trained recommendation model object (e.g., SVD algo, or functions for CB).
        test_data (pd.DataFrame): The test interaction data.
        train_data (pd.DataFrame): The train interaction data (needed to exclude already seen items).
        user_col (str): User ID column name.
        item_col (str): Item ID column name.
        k (int): Number of recommendations to evaluate.
        model_type (str): 'cf' for collaborative filtering (Surprise), 'cb' for content-based.
                         This determines how recommendations are generated. Requires adaptation.

    Returns:
        dict: Dictionary containing average precision@k and recall@k.
    """
    print(f"Evaluating model (type: {model_type})...")
    precisions = []
    recalls = []

    # Prepare ground truth: map each user to their relevant items in the test set
    test_user_items = test_data.groupby(user_col)[item_col].apply(set).to_dict()
    # Optional: Prepare train items to filter them out from recommendations
    # train_user_items = train_data.groupby(user_col)[item_col].apply(set).to_dict()

    # Get necessary components based on model type (NEEDS REFINEMENT based on actual model objects)
    if model_type == 'cf':
        algo, trainset, item_map = model # Assumes model tuple from collaborative_filtering.py
    elif model_type == 'cb':
         cosine_sim, item_indices, all_songs = model # Assumes model tuple from content_based.py
         # Need train interactions to find 'liked' songs for CB input
         train_user_items = train_data.groupby(user_col)[item_col].apply(list).to_dict()
    else:
         raise ValueError("Unsupported model_type for evaluation.")


    test_users = test_data[user_col].unique()
    print(f"Evaluating on {len(test_users)} users...")

    for user_id in test_users:
        true_relevant = test_user_items.get(user_id, set())
        if not true_relevant:
            continue # Skip users with no relevant items in test set

        # Generate recommendations for this user
        recommendations = []
        if model_type == 'cf':
            # Ensure item_map is passed correctly if needed by get_cf_recommendations
            recommendations = get_cf_recommendations(algo, user_id, trainset, item_map, top_n=k) # Use function from cf module
        elif model_type == 'cb':
            user_liked = train_user_items.get(user_id, [])
            if user_liked:
                 recommendations = get_content_based_recommendations(user_liked, cosine_sim, item_indices, all_songs, top_n=k) # Use function from cb module

        # Calculate metrics for this user
        p_at_k, r_at_k = precision_recall_at_k(recommendations, true_relevant, k)
        precisions.append(p_at_k)
        recalls.append(r_at_k)

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0

    print(f"Evaluation complete: Avg Precision@{k} = {avg_precision:.4f}, Avg Recall@{k} = {avg_recall:.4f}")
    return {'precision_at_k': avg_precision, 'recall_at_k': avg_recall}

# Helper functions from other modules needed for standalone execution if __name__ == '__main__'
# Add imports and potentially simplified versions if running this file directly.
# Note: Standalone evaluation example is complex due to dependencies on trained models.
# It's better integrated into a main script after models are trained.

# if __name__ == '__main__':
    # This part is tricky to run standalone as it needs trained models and data splits.
    # Example structure (requires models and data to be loaded/trained first):
    # print("Standalone evaluation example (requires trained models loaded):")
    # # 1. Load train_df, test_df
    # # 2. Load or train your CF model (algo, trainset, item_map) -> cf_model_components
    # # 3. Load or train your CB model (cosine_sim, item_indices, all_songs) -> cb_model_components
    # # 4. Call evaluate_model
    #
    # # cf_results = evaluate_model(cf_model_components, test_df, train_df, model_type='cf')
    # # print("CF Evaluation Results:", cf_results)
    # # cb_results = evaluate_model(cb_model_components, test_df, train_df, model_type='cb')
    # # print("CB Evaluation Results:", cb_results)