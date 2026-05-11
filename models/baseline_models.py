import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import argparse
import os

def load_data(data_dir="."):
    """Loads the train, validation, and test splits"""
    try:
        train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
        val_df = pd.read_csv(os.path.join(data_dir, "val_data.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test_data.csv"))
        return train_df, val_df, test_df
    except:
        print("Error, could not find CSV files.")
        exit(1)
    

def preprocess_features(train_df, val_df, test_df):
    """Separate featuers (X) and labels (y), and scales them"""
    feature_cols = ['x', 'y', 'displacement']
    target_col = 'is_bike_lane'

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # Standardizing the features 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled = scaler.transform(X_val.values)
    X_test_scaled = scaler.transform(X_test.values)
    return (X_train, y_train, X_train_scaled), (X_val, y_val, X_val_scaled), (X_test, y_test, X_test_scaled)

def logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test):
    """Linear classifier based on X, Y, and magnitude of displacement"""
    print("###### Training Logistic Regression ######")
    model = LogisticRegression(random_state=42, class_weight='balanced',)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))

def xgboost_classifier(X_train, y_train, X_test, y_test):
    """Tree-based classifier based on X, Y, and magnitude of displacement"""
    print("###### Training XGBoost Classifier ######")
    # calculate ratio of Class 0 examples to Class 1 examples
    model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))

def kmeans_clustering_magnitude(train_df):
    """Unsupervised clustering based on 1 second magnitude displacement"""
    print("###### Training KMeans Clustering #######")
    magnitudes = train_df[['displacement']].values
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10) # for now only looking at Slow/stopped vs Fast/moving
    train_df['speed_cluster'] = kmeans.fit_predict(magnitudes)
    cluster_centers = kmeans.cluster_centers_
    print(f"Cluster 0 Center (Avg Displacement): {cluster_centers[0][0]:.2f} meters/sec")
    print(f"Cluster 1 Center (Avg Displacement): {cluster_centers[1][0]:.2f} meters/sec")
    print("\nSample of clustered vehicles:")
    print(train_df[['track_id', 'displacement', 'speed_cluster']].head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline Classifiers")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing extracted CSV files")
    args = parser.parse_args()

    # load data
    train_df, val_df, test_df = load_data(args.data_dir)
    # prepare and scale features
    (X_train, y_train, X_train_scaled), (X_val, y_val, X_val_scaled), (X_test, y_test, X_test_scaled) = preprocess_features(train_df, val_df, test_df)
    # run the models
    logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)
    xgboost_classifier(X_train, y_train, X_test, y_test)
    kmeans_clustering_magnitude(train_df)
