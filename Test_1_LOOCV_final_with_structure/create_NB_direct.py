import pandas as pd
from sklearn.naive_bayes import ComplementNB
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
    """
    print(f"Training NB model for rep {rep_num}, percentage {percentage}, fold {fold_num}", flush=True)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Initialize encoder for the target variable (action)
    encoder = LabelEncoder()
    encoder.classes_ = np.array(['change_to_left', 'change_to_right', 'cruise', 'keep', 'swerve_left', 'swerve_right'])
    
    # Select ONLY the numerical features
    numerical_features = ["curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W", "latent_collision"]
    required_columns = ["action"] + numerical_features
    
    # Check if all required columns are present
    missing_columns = [col for col in required_columns if col not in train_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in training data: {missing_columns}")
    
    train_data = train_df[required_columns].copy()
    
    # DEBUG: Check data types and sample values
    print("Data types:", flush=True)
    for col in train_data.columns:
        print(f"  {col}: {train_data[col].dtype}, sample: {train_data[col].iloc[0] if len(train_data) > 0 else 'N/A'}", flush=True)
    
    # Convert boolean-like strings to numerical values
    for col in numerical_features:
        if train_data[col].dtype == 'object':
            print(f"Converting column '{col}' from string to numerical", flush=True)
            # Convert 'True'/'False' to 1/0
            train_data[col] = train_data[col].map({'True': 1, 'False': 0, 'true': 1, 'false': 0})
            # If there are any non-boolean values, try to convert to float
            try:
                train_data[col] = pd.to_numeric(train_data[col], errors='coerce')
            except:
                pass
    
    # Check for any remaining non-numerical values
    for col in numerical_features:
        if train_data[col].dtype == 'object':
            print(f"WARNING: Column '{col}' still contains non-numerical values after conversion", flush=True)
            print(f"Unique values in {col}: {train_data[col].unique()}", flush=True)
    
    # Prepare features and target
    X_train = train_data.drop(['action'], axis=1)
    y_train = encoder.fit_transform(train_data['action'])
    
    print(f"Feature matrix shape: {X_train.shape}", flush=True)
    print(f"Feature dtypes: {X_train.dtypes.to_dict()}", flush=True)
    print(f"Target distribution: {np.bincount(y_train)}", flush=True)
    print(f"Target classes: {encoder.classes_}", flush=True)
    
    # Check for any NaN values after conversion
    if X_train.isnull().any().any():
        print(f"WARNING: NaN values found in features after conversion. Filling with 0.", flush=True)
        X_train = X_train.fillna(0)
    
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
