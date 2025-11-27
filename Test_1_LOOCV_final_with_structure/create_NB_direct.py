import pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score
import pickle
import time
import statistics
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

def create_nb_model_from_data(train_df, percentage, rep_num, fold_num, model_path):
    """
    Create and save a Naive Bayes model from training data directly passed as DataFrame
    
    Parameters:
    train_df: pandas DataFrame with training data
    percentage: percentage string (e.g., "01", "25")
    rep_num: repetition number
    fold_num: fold number
    model_path: path where to save the trained model
    
    Returns:
    dict: Training results and metrics
    """
    print(f"Training NB model for rep {rep_num}, percentage {percentage}, fold {fold_num}", flush=True)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Initialize encoder
    encoder = LabelEncoder()
    encoder.classes_ = np.array(['change_to_left', 'change_to_right', 'cruise', 'keep', 'swerve_left', 'swerve_right'])
    
    # Select relevant features
    required_columns = ["action", "curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"]
    
    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in train_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in training data: {missing_columns}")
    
    train_data = train_df[required_columns].copy()
    
    # Prepare features and target
    X_train = train_data.drop(['action'], axis=1)
    y_train = encoder.transform(train_data['action'])
    
    # Define the hyperparameter grid for ComplementNB
    param_grid = {
        'alpha': [0.01],
        'fit_prior': [True]
    }
    
    # Manual hyperparameter evaluation
    best_f1_score = -1
    best_params = {}
    best_model = None
    
    training_times = []
    
    for alpha in param_grid['alpha']:
        for fit_prior in param_grid['fit_prior']:
            # Train model with current hyperparameters
            nb = ComplementNB(alpha=alpha, fit_prior=fit_prior)
            start_time = time.time()
            nb.fit(X_train, y_train)
            end_time = time.time()
            training_time = end_time - start_time
            training_times.append(training_time)
            
            # Use cross-validation for hyperparameter tuning
            cv_scores = cross_val_score(nb, X_train, y_train, cv=min(5, len(X_train)), scoring='f1_weighted')
            avg_f1 = np.mean(cv_scores)
            
            print(f"Fold {fold_num} - Params: alpha={alpha}, fit_prior={fit_prior}, CV F1-score: {avg_f1:.4f}", flush=True)
            
            # Update best parameters if this model is better
            if avg_f1 > best_f1_score:
                best_f1_score = avg_f1
                best_params = {'alpha': alpha, 'fit_prior': fit_prior, 'f1_score': avg_f1}
                best_model = nb
    
    print(f"Best parameters for fold {fold_num}: {best_params}", flush=True)
    print(f"Best CV F1-score for fold {fold_num}: {best_f1_score:.4f}", flush=True)
    
    # Save the best model with metadata
    model_data = {
        'model': best_model,
        'encoder': encoder,
        'feature_columns': X_train.columns.tolist(),
        'best_params': best_params
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    # Calculate training time statistics
    if training_times:
        avg_time = statistics.mean(training_times)
        stdev_time = statistics.stdev(training_times) if len(training_times) > 1 else 0
    else:
        avg_time = 0
        stdev_time = 0
    
    return {
        'fold': fold_num,
        'best_params': best_params,
        'best_f1_score': best_f1_score,
        'avg_training_time': avg_time,
        'stdev_training_time': stdev_time,
        'model_path': model_path
    }
