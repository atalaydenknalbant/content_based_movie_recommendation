# Content-Based Recommendation System

## Overview
This project implements a **content-based recommendation system** using matrix factorization and grid search optimization. It leverages PyTorch to build and train the model, focusing on hyperparameter tuning to achieve optimal performance.

## Key Features
- **Matrix Factorization Model**: Utilizes embeddings to learn latent factors for users and movies.
- **Grid Search**: Explores combinations of hyperparameters such as embedding size, learning rate, and regularization to find the best-performing model.
- **Visualization**: Analyzes the effect of hyperparameters on performance metrics using box plots.

## Dataset
The project uses the **MovieLens Small Dataset**, available at [GroupLens MovieLens Dataset](https://grouplens.org/datasets/movielens/latest/). This dataset contains:
- User and movie IDs
- Ratings
- Timestamps

## Requirements
- Python 3.8 or above
- Libraries:
  - PyTorch
  - pandas
  - NumPy
  - scikit-learn
  - tqdm
  - seaborn
  - matplotlib

## Project Workflow
1. **Data Preprocessing**:
   - Map user and movie IDs to unique indices.
   - Split the data into training and test sets.

2. **Model Definition**:
   - Build a PyTorch-based matrix factorization model with user and movie embeddings.
   - Use a fully connected layer for prediction.

3. **Hyperparameter Optimization**:
   - Perform grid search over the following hyperparameters:
     - Embedding size
     - Learning rate
     - Regularization
   - Evaluate the model using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

4. **Training and Evaluation**:
   - Train the model using SGD optimizer.
   - Measure performance on the test set after every training epoch.

5. **Results Visualization**:
   - Use box plots to analyze the effect of hyperparameters on test MAE.

## Usage
1. **Setup**:
   - Clone the repository and navigate to the project directory.
   - Install the required libraries using `pip install -r requirements.txt`.

2. **Run the Notebook**:
   - Open the `Content_Based_Recommendation.ipynb` notebook.
   - Execute the cells sequentially.

3. **Output**:
   - Top 10 configurations based on test MAE.
   - Visualizations for the impact of embedding size, learning rate, and regularization.

## Results
The results include:
- Optimal hyperparameter combinations.
- Detailed performance metrics (MAE, RMSE).
- Insights into how different hyperparameters affect the model's accuracy.
