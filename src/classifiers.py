import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.cluster import KMeans

def build_linear_nn_classifier(input_dim=2):
    """
    A simple Neural Network for coordinate classification (x, y).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        # Output layer for binary classification (bike lane vs not)
        tf.keras.layers.Dense(1, activation='sigmoid') 
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_xgboost_classifier():
    """
    XGBoost classifier for x, y coordinates.
    """
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss'
    )
    return model

def apply_kmeans_to_magnitude(magnitude_data, n_clusters=3):
    """
    Applies K-Means clustering specifically to the single magnitude displacement feature.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    # Reshape is required because it's a single feature
    clusters = kmeans.fit_predict(magnitude_data.reshape(-1, 1))
    return kmeans, clusters
