import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    """
    Loads training and test CSV files.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(train_df, test_df, features, target):
    """
    Preprocesses the data by handling missing values and splitting the training set.
    """
    X = train_df[features]
    y = train_df[target]
    
    # Handle missing values
    X = X.fillna(X.median())
    test_df[features] = test_df[features].fillna(test_df[features].median())
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_valid, y_train, y_valid, test_df
