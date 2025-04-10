
# Machine Learning Music Recommendation System: Hybrid Approach (Content & SVD) with Flask

This project is a music recommendation system built using the Last.fm dataset. The system utilizes machine learning algorithms to recommend songs to users based on their listening history and the characteristics of the music.

## Overview

This music recommendation system aims to provide personalized music suggestions to users based on their past interactions with songs. It employs two main types of recommendation algorithms:

* **Content-Based Recommendation:** This approach recommends music to users based on the attributes of the songs they have liked in the past. It analyzes features such as the artist, album, and track name to find similar music. If a user enjoys songs by a particular artist or in a specific genre (inferred from the metadata), the system will recommend other songs with similar characteristics.

* **Collaborative Filtering (SVD):** This approach leverages the listening behavior of a community of users to make recommendations. By finding users with similar listening patterns, the system can recommend songs that users with similar tastes have enjoyed, even if those songs have different content features. This implementation uses Singular Value Decomposition (SVD), a matrix factorization technique, to uncover latent relationships in user-song interactions.

The system consists of the following main components:

1.  **Data Loading and Preprocessing:** This step involves loading the Last.fm dataset, cleaning the data, and preparing it for analysis. This includes handling missing values, creating unique identifiers for songs, and filtering out sparse user and item interactions.

2.  **Model Training:** In this step, both the Content-Based and Collaborative Filtering (SVD) models are trained on the preprocessed data. The Content-Based model builds a similarity matrix based on song metadata, while the SVD model learns latent factors from user-song interaction patterns.

3.  **Recommendation Generation (Flask Application):** The trained models are integrated into a Flask web application. Users can select their username from a dropdown list, and the system will generate personalized music recommendations based on their historical listening data using both recommendation approaches.

## Dataset

The project uses the Last.fm dataset, which contains user listening history and song metadata. The dataset can be downloaded from [here](https://www.kaggle.com/datasets/harshal19t/lastfm-dataset/).

## File Structure

The project has the following file structure:
```
├── app.py                      # Flask application to run the web interface
├── collaborative_filtering.py  # Implementation of the SVD collaborative filtering model
├── config.py                   # Configuration settings for data paths, model parameters, etc.
├── content_based.py            # Implementation of the content-based filtering model
├── data_loader.py              # Functions to load and initially process the Last.fm data
├── eda.py                      # Functions for exploratory data analysis (optional)
├── evaluation.py               # Functions to evaluate the performance of the recommendation models
├── main.py                     # Main script to run the entire recommendation pipeline (training and example recommendations)
├── preprocessing.py            # Functions for data cleaning, feature engineering, and splitting
├── requirements.txt            # Lists the Python dependencies for the project
├── lastfm_data.csv             # The Last.fm dataset (ensure the path in config.py is correct)
├── templates/
│   ├── index.html              # HTML template for the homepage with user selection
│   └── recommendations.html  # HTML template to display the music recommendations
└── .gitignore                  # Specifies intentionally untracked files that Git should ignore
```

## How to Use

The system has been tested on Python versions up to 3.11. When running it with higher Python versions, you may encounter errors due to compatibility issues with some pip packages.

To use the system, follow these steps:

1.  **Clone the repository** (if you haven't already).
2.  **Create a virtual environment** (using `venv` or `virtualenv`) in the project directory.
3.  **Activate the virtual environment.**
4.  **Install the required dependencies.** Run `pip install -r requirements.txt`.
5.  **(Optional) Train the models using the standalone script.** You can run the `main.py` script to load data, preprocess it, train both recommendation models, evaluate them, and print example recommendations to the console. Run `python main.py`. This step is optional as the Flask app also trains the models on startup.
6.  **Run the Flask application.** Execute the `app.py` file to start the web interface. Run `python app.py`.
7.  **Open your web browser** and navigate to `http://127.0.0.1:5000/` (or the address where your Flask app is running).
8.  **On the homepage, you will see a dropdown list of available users.** Select any username from the list.
9.  **Click the "Get Recommendations" button.**
10. **You will be redirected to the recommendations page**, which will display music recommendations generated by both the SVD and Content-Based models for the selected user.

## Dependencies

The system requires the following dependencies, which are listed in the `requirements.txt` file:

* Flask
* pandas
* matplotlib
* numpy
* seaborn
* scikit-learn
* scipy
* surprise (scikit-surprise)

## Screenshots

Here are some screenshots of the system:

**Homepage with User Selection**
![Homepage with User Selection](screenshots/index-output.png)

**Recommendations Page**
![Recommendations Page](screenshots/recommendation-output.png)

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Visit and Follow
For more details and tutorials, visit the website: [DocsAllOver](https://docsallover.com/).

Follow us on:
- [Facebook](https://www.facebook.com/docsallover)
- [Instagram](https://www.instagram.com/docsallover.tech/)
- [x.com](https://www.x.com/docsallover/)
- [LinkedIn](https://www.linkedin.com/company/docsallover/)
- [YouTube](https://www.youtube.com/@docsallover)
- [Threads.net](https://threads.net/docsallover.tech)

and visit our website to know more about our tutorials and blogs.