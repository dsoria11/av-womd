import pandas as pd
from data_preprocessing import calculate_magnitude_displacement, split_data_by_object, prepare_coordinate_features
from classifiers import build_linear_nn_classifier, build_xgboost_classifier, apply_kmeans_to_magnitude

def main():
    print("Loading data...")
    # NOTE: Replace 'your_womd_data.csv' with your actual data loading mechanism
    # df = pd.read_csv('your_womd_data.csv')
    
    # --- Dummy data for demonstration purposes ---
    df = pd.DataFrame({
        'object_id': [1]*20 + [2]*20 + [3]*20 + [4]*20 + [5]*20,
        'x': range(100),
        'y': range(100, 200),
        'is_bike_lane': [0, 1] * 50
    })
    
    print("Extracting magnitude displacement (1s window)...")
    df = calculate_magnitude_displacement(df, time_steps_1s=10)
    
    print("Splitting data (60/20/20) securely by object_id...")
    train_df, val_df, test_df = split_data_by_object(df, 'object_id')
    
    print(f"Train objects: {train_df['object_id'].nunique()}, Val objects: {val_df['object_id'].nunique()}, Test objects: {test_df['object_id'].nunique()}")
    
    # Prepare X, Y features for bike lane classification
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_coordinate_features(train_df, val_df, test_df)
    
    print("\n--- Training Neural Network ---")
    nn_model = build_linear_nn_classifier(input_dim=2)
    nn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)
    loss, accuracy = nn_model.evaluate(X_test, y_test, verbose=0)
    print(f"NN Test Accuracy: {accuracy:.4f}")
    
    print("\n--- Training XGBoost ---")
    xgb_model = build_xgboost_classifier()
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_score = xgb_model.score(X_test, y_test)
    print(f"XGBoost Test Accuracy: {xgb_score:.4f}")
    
    print("\n--- Applying K-Means to Magnitude Displacement ---")
    # Extract just the magnitude column for the training set
    mag_train = train_df['magnitude_displacement_1s'].values
    kmeans_model, train_clusters = apply_kmeans_to_magnitude(mag_train, n_clusters=3)
    print(f"Assigned {len(train_clusters)} magnitude records to 3 clusters.")

if __name__ == "__main__":
    main()
