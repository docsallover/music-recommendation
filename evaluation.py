# evaluation.py
"""
Functions to evaluate recommendation models using ranking metrics (Precision@k, Recall@k).
Adapted for implicit feedback scenario.
"""
import numpy as np
import pandas as pd # Import pandas for DataFrame handling
import config
# Need recommendation functions from other modules if running evaluation directly
from collaborative_filtering import get_cf_recommendations_surprise, get_als_recommendations
import collaborative_filtering
from content_based import get_content_based_recommendations


def precision_recall_at_k(predictions, true_relevant_items, k=config.N_RECOMMENDATIONS):
    """ Calculates Precision@k and Recall@k for a single user. """
    if k <= 0: return 0.0, 0.0
    predictions_at_k = predictions[:k] # Ensure we only look at top k preds
    true_relevant_items_set = set(true_relevant_items) # Ensure it's a set for efficient lookup
    relevant_in_predictions = set(predictions_at_k) & true_relevant_items_set
    num_relevant_in_predictions = len(relevant_in_predictions)

    precision = num_relevant_in_predictions / len(predictions_at_k) if len(predictions_at_k) > 0 else 0.0
    recall = num_relevant_in_predictions / len(true_relevant_items_set) if len(true_relevant_items_set) > 0 else 0.0

    return precision, recall

def evaluate_model(model_components, test_data, train_data, # Pass train_data for filtering history
                   user_col='user_id', item_col='item_id',
                   k=config.N_RECOMMENDATIONS, model_type='svd'):
    """
    Evaluates a recommendation model over a test set for implicit feedback.

    Args:
        model_components: Tuple containing necessary trained model objects and mappings.
                          Format depends on model_type.
                          - 'svd': (surprise_algo, surprise_trainset, item_map)
                          - 'als': (implicit_model, sparse_user_item_matrix_train, user_map_train, item_map_train)
                          - 'cb': (cosine_sim_matrix, item_indices_map, all_items_set)
        test_data (pd.DataFrame): The test interaction data.
        train_data (pd.DataFrame): The train interaction data (needed for CB history and potentially filtering).
        user_col (str): User ID column name.
        item_col (str): Item ID column name.
        k (int): Number of recommendations to evaluate.
        model_type (str): Type of model ('svd', 'als', 'cb').

    Returns:
        dict: Dictionary containing average precision@k and recall@k.
    """
    print(f"--- Evaluating model (type: {model_type}) ---")
    precisions = []
    recalls = []

    # Prepare ground truth: map each user to their relevant items in the test set
    test_user_items = test_data.groupby(user_col)[item_col].apply(set).to_dict()

    # Prepare training history (needed for CB, potentially others)
    train_user_items_history = train_data.groupby(user_col)[item_col].apply(list).to_dict()

    # Unpack model components based on type
    if model_type == 'svd':
        algo, trainset, item_map_eval = model_components
    elif model_type == 'als':
        if not collaborative_filtering.IMPLICIT_AVAILABLE:
             print("Skipping ALS evaluation: 'implicit' library not available.")
             return {'precision_at_k': 0, 'recall_at_k': 0}
        als_model, sparse_matrix_train, user_map_train, item_map_eval = model_components
    elif model_type == 'cb':
        cosine_sim, item_indices, all_items_set = model_components
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for evaluation.")

    test_users = list(test_user_items.keys()) # Evaluate only users present in test set
    print(f"Evaluating on {len(test_users)} users from the test set...")
    evaluated_users = 0

    for user_id in test_users:
        true_relevant = test_user_items.get(user_id, set())
        if not true_relevant:
            continue # Skip users with no relevant items in test set ground truth

        recommendations = []
        try:
            if model_type == 'svd':
                # item_map_eval should map Surprise internal IDs -> original item_ids
                recommendations = get_cf_recommendations_surprise(algo, user_id, trainset, item_map_eval, top_n=k)
            elif model_type == 'als':
                 # Need user index for ALS
                 if user_id in user_map_train['user_id'].values:
                      user_idx = user_map_train[user_map_train['user_id'] == user_id].index[0]

                      # <<< --- ADD THIS CHECK --- >>>
                      num_users_in_matrix = sparse_matrix_train.shape[0]
                      if user_idx >= num_users_in_matrix:
                           print(f"DEBUG EVALUATION ERROR: User ID {user_id} maps to index {user_idx}, but ALS matrix only has {num_users_in_matrix} users (indices 0-{num_users_in_matrix-1}). Skipping user.")
                           continue # Skip this user - index is invalid!
                      # <<< --- END CHECK --- >>>

                      # item_map_eval maps internal ALS index -> original item_ids
                      recommendations = get_als_recommendations(als_model, user_idx, sparse_matrix_train, item_map_eval, top_n=k)
                 else:
                      # User might be in test but filtered out during sparse matrix creation for train
                      # print(f"DEBUG: User {user_id} from test set not found in ALS training user map. Skipping.") # Optional debug print
                      continue # Skip user if not in ALS model's user map
            elif model_type == 'cb':
                user_liked = train_user_items_history.get(user_id, [])
                if user_liked:
                     # item_indices maps original item_ids -> similarity matrix index
                     # all_items_set contains all valid original item_ids
                     recommendations = get_content_based_recommendations(user_liked, cosine_sim, item_indices, all_items_set, top_n=k)

        except Exception as e:
            print(f"Error getting recommendations for user {user_id} ({model_type}): {e}")
            continue # Skip user if recommendation fails


        # Calculate metrics for this user
        p_at_k, r_at_k = precision_recall_at_k(recommendations, true_relevant, k)
        precisions.append(p_at_k)
        recalls.append(r_at_k)
        evaluated_users += 1

        if evaluated_users % 500 == 0: # Print progress
             print(f"Evaluated {evaluated_users}/{len(test_users)} users...")

    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0

    print(f"Evaluation complete for {evaluated_users} users.")
    print(f"Avg Precision@{k} = {avg_precision:.4f}")
    print(f"Avg Recall@{k} = {avg_recall:.4f}")
    print("-----------------------------")
    return {'precision_at_k': avg_precision, 'recall_at_k': avg_recall}


# if __name__ == '__main__':
    # Standalone execution is complex: requires loading pre-split data,
    # training models, and having all components ready.
    # Better to run evaluation from the main script.
    # print("Evaluation module standalone execution needs pre-trained models and data.")