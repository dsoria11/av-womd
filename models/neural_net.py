import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
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
    # feature_cols = ['x', 'y', 'displacement']
    feature_cols = ['displacement', 'dist_to_bike_lane'] 
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
        tf.keras.layers.InputLayer(input_shape=(2,)), # input layer for features
        tf.keras.layers.Dense(128, activation='relu'), # 1st hidden layer 
        tf.keras.layers.BatchNormalization(), # added batch normalization
        tf.keras.layers.Dropout(0.3), # added dropout to prevent overfitting
        tf.keras.layers.Dense(64, activation='relu'), # 2nd hidden layer
        tf.keras.layers.BatchNormalization(), # added batch normalization
        tf.keras.layers.Dense(1, activation='sigmoid'), # binary output (0 or 1)
    ])

    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    ) 

    # train the model 
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True
    )
    print("###### Training the model ######\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=256,
        class_weight=class_weight,
        callbacks=[lr_scheduler, early_stopping],
        verbose=1
    )

    print("\n###### Generating Diagnostic Plots ######\n")
    
    # ==========================================
    # FIGURE 1: Training History (Loss, Accuracy, AUC)
    # ==========================================
    plt.figure(figsize=(18, 5))
    
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
    plt.title('Loss Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
    plt.title('Accuracy Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # AUC (Area Under Curve)
    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'], label='Train AUC', color='blue')
    plt.plot(history.history['val_auc'], label='Val AUC', color='orange')
    plt.title('AUC Over Time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("1_training_history.png")
    plt.close()

    # ==========================================
    # FIGURE 2: Model Diagnostics
    # ==========================================
    # Flatten the predictions to a 1D array for easier math and plotting
    raw_predictions = model.predict(X_test, verbose=0).flatten()
    plt.figure(figsize=(16, 12))

    # Plot A: Prediction Score Distribution
    plt.subplot(2, 2, 1)
    plt.hist(raw_predictions[y_test == 0], bins=50, alpha=0.5, color='blue', label='Actual: Not Bike Lane', density=True)
    plt.hist(raw_predictions[y_test == 1], bins=50, alpha=0.5, color='orange', label='Actual: Bike Lane', density=True)
    plt.title('Prediction Score Distribution')
    plt.xlabel('Predicted Probability (0.0 to 1.0)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot B: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, raw_predictions)
    roc_auc = auc(fpr, tpr)
    plt.subplot(2, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot C: Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, raw_predictions)
    ap = average_precision_score(y_test, raw_predictions)
    plt.subplot(2, 2, 3)
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (Avg Precision = {ap:.3f})')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # --- AUTOMATIC THRESHOLD CALCULATION ---
    # Calculates the F1 score for every possible threshold to find the mathematical peak
    fscore = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
    best_idx = np.argmax(fscore)
    best_thresh = thresholds[best_idx]
    best_f1 = fscore[best_idx]
    
    print(f"Optimal Threshold mathematically derived from PR Curve: {best_thresh:.3f} (F1-Score: {best_f1:.3f})")

    # Plot D: Confusion Matrix at the Best Threshold
    best_preds = (raw_predictions > best_thresh).astype(int)
    cm = confusion_matrix(y_test, best_preds)
    ax = plt.subplot(2, 2, 4)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal Road', 'Bike Lane'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(f'Confusion Matrix (using {best_thresh:.2f} threshold)')

    plt.tight_layout()
    plt.savefig("2_model_diagnostics.png")
    plt.close()
    
    print("-> Saved diagnostic graphs to: 1_training_history.png and 2_model_diagnostics.png\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network to classify bike lanes.")
    parser.add_argument("--data_dir", type=str, default="feature_extraction", help="Directory containing train_data.csv, val_data.csv, and test_data.csv")
    args = parser.parse_args()

    train_df, val_df, test_df = load_data(args.data_dir)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_features(train_df, val_df, test_df)
    build_train_nn(X_train, y_train, X_val, y_val, X_test, y_test)
