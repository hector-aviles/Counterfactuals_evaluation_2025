#!/usr/bin/env python3
import pandas as pd
from sklearn.naive_bayes import ComplementNB
import pickle
import time
import statistics
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train Naive Bayes model from training file')
    parser.add_argument('--train_file', type=str, required=True, help='Path to training data CSV file')
    parser.add_argument('--percentage', type=str, required=True, help='Percentage string')
    parser.add_argument('--rep_num', type=int, required=True, help='Repetition number')
    parser.add_argument('--fold_num', type=int, required=True, help='Fold number')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save the trained model')
    
    args = parser.parse_args()
    
    print(f"Training NB model for rep {args.rep_num}, percentage {args.percentage}, fold {args.fold_num}", flush=True)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Load training data
    train_df = pd.read_csv(args.train_file)
    
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
            
            print(f"Fold {args.fold_num} - Params: alpha={alpha}, fit_prior={fit_prior}, CV F1-score: {avg_f1:.4f}", flush=True)
            
            # Update best parameters if this model is better
            if avg_f1 > best_f1_score:
                best_f1_score = avg_f1
                best_params = {'alpha': alpha, 'fit_prior': fit_prior, 'f1_score': avg_f1}
                best_model = nb
    
    print(f"Best parameters for fold {args.fold_num}: {best_params}", flush=True)
    print(f"Best CV F1-score for fold {args.fold_num}: {best_f1_score:.4f}", flush=True)
    
    # Save the best model with metadata
    model_data = {
        'model': best_model,
        'encoder': encoder,
        'feature_columns': X_train.columns.tolist(),
        'best_params': best_params
    }
    
    with open(args.model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {args.model_path}", flush=True)
    
    # Calculate training time statistics
    if training_times:
        avg_time = statistics.mean(training_times)
        stdev_time = statistics.stdev(training_times) if len(training_times) > 1 else 0
    else:
        avg_time = 0
        stdev_time = 0
    
    print(f"Average training time: {avg_time:.4f} seconds", flush=True)
    print(f"Training time stdev: {stdev_time:.4f} seconds", flush=True)

if __name__ == "__main__":
    main()