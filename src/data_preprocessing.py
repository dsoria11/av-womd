import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def calculate_magnitude_displacement(df, time_steps_1s=10):
    """
    Calculates the magnitude displacement over a 1-second window.
    Data is sampled at 10Hz (10 steps = 1 second) and sorted by time.
    """
    # Group by object to ensure we don't calculate displacement across different vehicles
    df = df.copy()
    
    # Calculate differences over the time window
    df['dx'] = df.groupby('object_id')['x'].diff(periods=time_steps_1s)
    df['dy'] = df.groupby('object_id')['y'].diff(periods=time_steps_1s)
    
    # Calculate magnitude: sqrt(dx^2 + dy^2)
    df['magnitude_displacement_1s'] = np.sqrt(df['dx']**2 + df['dy']**2)
    
    # Drop rows where displacement couldn't be calculated (first 1s of data for each object)
    return df.dropna(subset=['magnitude_displacement_1s'])

def split_data_by_object(df, object_id_col='object_id'):
    """
    Splits data 60% Train, 20% Val, 20% Test, ensuring no object overlaps between sets.
    """
    # First split: 60% Train, 40% Temp (Val + Test)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.60, random_state=42)
    train_idx, temp_idx = next(gss1.split(df, groups=df[object_id_col]))
    
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    
    # Second split: Split the remaining 40% into 50/50 (which equals 20% Val / 20% Test overall)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[object_id_col]))
    
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]
    
    return train_df, val_df, test_df

def prepare_coordinate_features(train_df, val_df, test_df):
    """
    Extracts purely X, Y coordinates for the neural network/XGBoost inputs
    and the bike lane classification targets.
    """
    features = ['x', 'y']
    target = 'is_bike_lane' # Assumes this binary column exists (1 = yes, 0 = no)
    
    X_train, y_train = train_df[features].values, train_df[target].values
    X_val, y_val = val_df[features].values, val_df[target].values
    X_test, y_test = test_df[features].values, test_df[target].values
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
