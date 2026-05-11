import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import argparse
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to hide CUDA warnings

def load_data(data_dir="."):
    """Loads the train, validation, and test splits"""
    try:
        train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
        val_df = pd.read_csv(os.path.join(data_dir, "val_data.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test_data.csv"))
        return train_df, val_df, test_df
    except FileNotFoundError:
        print("Error: Could not find CSV files.")
        exit(1)

def prepare_features(train_df, val_df, test_df):
    """Prepares and strictly scales coordinates for the neural network"""
    feature_cols = ['x', 'y', 'displacement']
    target_col = 'is_bike_lane'

    X_train, y_train = train_df[feature_cols].values, train_df[target_col].values
    X_val, y_val = val_df[feature_cols].values, val_df[target_col].values
    X_test, y_test = test_df[feature_cols].values, test_df[target_col].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)

def build_train_nn(X_train, y_train, X_val, y_val, X_test, y_test):
    """Builds a dense neural network """
    print("Building and training the neural network...\n")
    neg_count = np.sum(y_train == 0) # count of negative samples
    pos_count = np.sum(y_train == 1) # count of positive samples
    total = len(y_train)
    weight_for_0 = (1 / neg_count) * (total / 2.0)
    weight_for_1 = (1 / pos_count) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    # class_weight = {0: 1.0, 1: 10.0} # manually set to handle imbalance

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(3,)), # input layer
        tf.keras.layers.Dense(256, activation='relu'), # 1st hidden layer 
        tf.keras.layers.Dropout(0.2), # adjusted dropout to prevent overfitting
        tf.keras.layers.Dense(128, activation='relu'), # 2nd hidden layer 
        tf.keras.layers.Dense(1, activation='sigmoid'), # binary output (0 or 1)
    ])

    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    ) 

    # train the model 
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    print("###### Training the model ######\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10000,
        batch_size=256,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=1
    )

    plt.figure(figsize=(12, 5))
    # Plot 1 Loss over time
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot 2 Accuracy over time
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_filename = "nn_training_history.png"
    plt.savefig(plot_filename)
    print(f"-> Saved training graphs to: {plot_filename}\n")

    # evaluate the model
    # print("\n###### Neural Network Evaluation ######\n")
    # raw_predictions = model.predict(X_test, verbose=0)
    # predictions = (raw_predictions > 0.9).astype(int) #.flatten()
    # print("Accuracy: ", accuracy_score(y_test, predictions))
    # print(classification_report(y_test, predictions, zero_division=0))

    # evaluate the model
    print("\n###### Neural Network Evaluation (Threshold Sweep) ######\n")
    raw_predictions = model.predict(X_test, verbose=0)
    
    # Test a range of thresholds from 0.50 to 0.85 without retraining
    thresholds_to_test = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    
    for t in thresholds_to_test:
        print(f"--- Threshold: {t} ---")
        predictions = (raw_predictions > t).astype(int)
        
        # Extract just the Class 1 metrics to keep the terminal readable
        report = classification_report(y_test, predictions, zero_division=0, output_dict=True)
        
        if '1' in report:
            p = report['1']['precision']
            r = report['1']['recall']
            f1 = report['1']['f1-score']
            print(f"Class 1 -> Precision: {p:.3f} | Recall: {r:.3f} | F1-Score: {f1:.3f}\n")
        else:
            print("Class 1 -> Precision: 0.000 | Recall: 0.000 | F1-Score: 0.000\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network to classify bike lanes.")
    parser.add_argument("--data_dir", type=str, default="feature_extraction", help="Directory containing train_data.csv, val_data.csv, and test_data.csv")
    args = parser.parse_args()

    train_df, val_df, test_df = load_data(args.data_dir)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_features(train_df, val_df, test_df)
    build_train_nn(X_train, y_train, X_val, y_val, X_test, y_test)
