
# MovieLens Content-Based Recommendation System

This project implements a **Content-Based Recommendation System** using the MovieLens dataset. It incorporates grid search for hyperparameter tuning and uses advanced techniques like embeddings, cosine similarity, and pre-trained language models to provide movie recommendations and predict ratings.

## Features
- **Content-Based Recommendation**: Combines movie tags and plot summaries to create a comprehensive feature set for recommendations.
- **Embedding-Based Similarity**: Utilizes pre-trained models to generate movie embeddings for accurate similarity computation.
- **MAE Calculation**: Evaluates the accuracy of predicted ratings using Mean Absolute Error.
- **Top-N Recommendations**: Provides top-N movie recommendations for users and calculates hit ratios.
- **GPU Acceleration**: Uses PyTorch for efficient computations on GPU.
- **Hyperparameter Tuning**: Implements grid search to optimize parameters like embedding size, learning rate, and regularization.

## Dataset
The MovieLens dataset is sourced from [GroupLens](https://grouplens.org/datasets/movielens/latest/). It includes various files like `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`

## Usage

### 1. Preprocessing
- **Dataset Preparation**: Merge tags and plot summaries into a single feature column.
- **Cleaning**: Normalize and clean text data.
- **Embedding Generation**: Create movie embeddings using a pre-trained language model.

### 2. Model Training and Evaluation
- **Grid Search**: Optimize hyperparameters for the recommendation system.
- **Rating Prediction**: Predict user ratings for movies and evaluate using Mean Absolute Error (MAE).
- **Recommendation Evaluation**: Generate Top-N recommendations and calculate the hit ratio.

### 3. Running the Notebooks
- **Grid Search Notebook**: Tune hyperparameters for optimal performance.
- **Content-Based Recommendation Notebook**: Train, evaluate, and generate recommendations.

```bash
jupyter notebook grid_search.ipynb
jupyter notebook Content_Based_Recommendation.ipynb
```

## Results
- **MAE**: Achieved an average MAE of `0.66` on the rating prediction task.
- **Hit Ratio**: Evaluated hit ratios for Top-N recommendations (N=10) with a robust evaluation setup.

## Directory Structure
```
movielens-recommender/
├── data/                     # Place MovieLens dataset files here
├── grid_search.ipynb         # Hyperparameter tuning with grid search
├── Content_Based_Recommendation.ipynb  # Content-based recommendation system
├── requirements.txt          # Python dependencies
└── README.md                 # Project description
```

## Dependencies
- Python 3.8+
- Pandas, NumPy
- PyTorch
- SentenceTransformers
- scikit-learn
- tqdm




