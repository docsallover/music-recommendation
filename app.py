# app.py
from flask import Flask, render_template, request, redirect, url_for
import main  # Import the main script

app = Flask(__name__)

# Global variables to store the trained models and data
trained_models = {}
train_data_components = {}
pipeline_initialized = False

def initialize_pipeline():
    """Runs the main pipeline once to load models."""
    print("Initializing pipeline and training models...")
    global trained_models
    global train_data_components
    global pipeline_initialized

    # Capture the return values from a modified run_pipeline
    results = run_pipeline_for_flask()
    if results:
        trained_models['cb_model'] = results.get('cb_model')
        trained_models['svd_model'] = results.get('svd_model')
        train_data_components['train_df'] = results.get('train_df')
        train_data_components['train_user_map'] = results.get('train_user_map')
        train_data_components['train_item_map'] = results.get('train_item_map')
        print("Pipeline initialization complete.")
        pipeline_initialized = True
    else:
        print("Pipeline initialization failed.")
        pipeline_initialized = True # Set to True to avoid repeated failures

def run_pipeline_for_flask():
    """
    Executes the main steps of the recommendation system project and returns trained models.
    Modified for Flask integration.
    """
    print("--- Starting Last.fm Music Recommendation Pipeline (for Flask) ---")

    # 1. Load Data
    print("\n--- 1. Loading Data ---")
    df_interactions = main.data_loader.load_lastfm_data()
    if df_interactions is None or df_interactions.empty:
        print("Failed to load interaction data or data is empty. Exiting.")
        return None

    # 2. Feature Engineering: Create Item ID
    print("\n--- 2. Creating Item IDs ---")
    df_interactions = main.preprocessing.create_item_id(df_interactions)

    # 4. Preprocessing: Filtering & Splitting (using the whole dataset for training in this Flask app for simplicity)
    print("\n--- 4. Filtering Sparse Data ---")
    df_filtered = main.preprocessing.filter_sparse_data(df_interactions)
    if df_filtered.empty:
        print("DEBUG Flask: df_filtered is empty! Check filtering/data.")
        return None

    train_df = df_filtered.copy()
    print(f"DEBUG Flask: train_df shape: {train_df.shape}")
    print(f"DEBUG Flask: Unique users in Flask's train_df: {train_df['user_id'].nunique()}")
    print(f"DEBUG Flask: First few rows of Flask's train_df:\n{train_df.head()}")

    _, train_user_map, train_item_map, _ = main.preprocessing.create_user_item_matrix(train_df.copy())
    print(f"DEBUG Flask: train_user_map:\n{train_user_map.head()}")
    # --- Prepare components needed for different models ---
    print("\n--- Preparing Training Set Components ---")
    train_item_metadata = main.content_based.get_item_metadata_df(train_df)
    all_items_in_train = set(train_item_metadata['item_id'])

    # 5. Train Models
    print("\n--- 5. Training Models ---")

    # Content-Based Model
    cb_model_components = None
    if not train_item_metadata.empty:
        try:
            print("Training Content-Based Model...")
            cb_feature_cols = ['artist', 'album', 'track'] # Choose features
            cb_similarity_matrix, cb_indices = main.content_based.build_content_similarity_matrix(train_item_metadata, feature_cols=cb_feature_cols)
            cb_model_components = (cb_similarity_matrix, cb_indices, all_items_in_train)
            print("Content-Based similarity matrix built.")
        except Exception as e:
            print(f"Error building Content-Based model: {e}")
    else:
        print("Skipping Content-Based model training (no metadata extracted from train set).")

    # Collaborative Filtering: SVD (Surprise)
    svd_model_components = None
    try:
        print("Training CF Model (SVD)...")
        svd_algo, svd_trainset = main.collaborative_filtering.train_svd_model_implicit(train_df)
        svd_model_components = (svd_algo, svd_trainset, train_item_map) # Use item_map derived from train_df
        print("SVD model trained.")
    except Exception as e:
        print(f"Error training SVD model: {e}")

    return {
        'cb_model': cb_model_components,
        'svd_model': svd_model_components,
        'train_df': train_df,
        'train_user_map': train_user_map,
        'train_item_map': train_item_map
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    global pipeline_initialized
    if not pipeline_initialized:
        initialize_pipeline()
    if request.method == 'POST':
        username = request.form['username']
        return redirect(url_for('recommendations', username=username))
    return render_template('index.html')


@app.route('/recommendations', methods=['GET'])
def recommendations():
    """Generates and displays music recommendations for a given username."""
    username = request.args.get('username')
    svd_recommendations = []
    cb_recommendations = []

    print(f"DEBUG Flask: /recommendations route was hit!")
    print(f"DEBUG Flask: Username in /recommendations: {username}")
    print(f"DEBUG Flask: trained_models is not None: {trained_models is not None}")
    print(f"DEBUG Flask: train_data_components is not None: {train_data_components is not None}")

    if username and trained_models and train_data_components:
        print(f"DEBUG Flask: Contents of trained_models: {trained_models.keys()}")
        print(f"DEBUG Flask: Contents of train_data_components: {train_data_components.keys()}")

        train_df = train_data_components.get('train_df')
        print(f"DEBUG Flask: train_df is None: {train_df is None}")
        if train_df is not None:
            print(f"DEBUG Flask: train_df is empty: {train_df.empty}")
            user_interactions = train_df[train_df['user_id'] == username]
            print(f"Number of interactions for user '{username}' in training data: {len(user_interactions)}")
        else:
            print("DEBUG Flask: train_df is None inside if block!")

        # --- SVD Recommendations ---
        svd_model_components = trained_models.get('svd_model')
        print(f"DEBUG Flask: svd_model_components is None: {svd_model_components is None}")
        train_item_map = train_data_components.get('train_item_map')
        print(f"DEBUG Flask: train_item_map is None: {train_item_map is None}")

        if svd_model_components and (not hasattr(train_item_map, 'empty') or not train_item_map.empty):
            try:
                svd_recs = main.collaborative_filtering.get_cf_recommendations_surprise(
                    svd_model_components[0], username, svd_model_components[1], svd_model_components[2] # algo, trainset, item_map
                )
                if svd_recs:
                    svd_recommendations = svd_recs # Directly assign the list
                else:
                    svd_recommendations = ["No SVD recommendations available."]
            except Exception as e:
                svd_recommendations = [f"Error generating SVD recommendations: {e}"]
        else:
            svd_recommendations = ["SVD model not available."]

        # --- Content-Based Recommendations ---
        cb_model_components = trained_models.get('cb_model')
        print(f"DEBUG Flask: cb_model_components is None: {cb_model_components is None}")
        train_df = train_data_components.get('train_df') # Getting train_df again - this is redundant but okay for now
        if cb_model_components and not train_df.empty:
            try:
                user_liked_items = train_df[train_df['user_id'] == username]['item_id'].unique().tolist()
                print(f"DEBUG Flask: user_liked_items for '{username}': {user_liked_items}")
                if user_liked_items:
                    cb_recs = main.content_based.get_content_based_recommendations(
                        user_liked_items, cb_model_components[0], cb_model_components[1], cb_model_components[2] # sim_matrix, indices, all_items
                    )
                    if cb_recs:
                        cb_recommendations = cb_recs # Directly assign the list
                    else:
                        cb_recommendations = ["No content-based recommendations generated for this user."]
                else:
                    cb_recommendations = ["User has no known history in training data for content-based recommendations."]
            except Exception as e:
                cb_recommendations = [f"Error generating content-based recommendations: {e}"]
        else:
            cb_recommendations = ["Content-based model not available."]

    else:
        return "Error: Models or data not loaded properly."

    return render_template('recommendations.html', username=username, svd_recommendations=svd_recommendations, cb_recommendations=cb_recommendations)

if __name__ == '__main__':
    app.run(debug=True)