import numpy as np
import pandas as pd
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================
# ACTION DEFINITIONS (from README)
# Used to label clusters after fitting, not during training.
# ============================================================
ACTION_LABELS = {
    'safe_cruising':      {'speed_min': 5.0,  'accel_min': -1.5, 'accel_max': 2.0,  'heading_max': 10.0},
    'aggressive_driving': {'speed_min': 5.0,  'accel_min': 2.0,  'accel_max': 99.0, 'heading_max': 10.0},
    'smooth_stop':        {'speed_min': 0.0,  'accel_min': -8.0, 'accel_max': -1.0, 'heading_max': 10.0},
    'emergency_braking':  {'speed_min': 0.0,  'accel_min': -99.0,'accel_max': -8.0, 'heading_max': 10.0},
    'intersection_turn':  {'speed_min': 0.0,  'accel_min': -99.0,'accel_max': 99.0, 'heading_max': 999.0, 'heading_min': 45.0},
    'parked_or_stopped':  {'speed_min': 0.0,  'speed_max': 1.0,  'accel_min': -1.5, 'accel_max': 1.5,   'heading_max': 10.0},
}

ACTION_COLORS = {
    'safe_cruising':      'steelblue',
    'aggressive_driving': 'crimson',
    'smooth_stop':        'gold',
    'emergency_braking':  'darkorange',
    'intersection_turn':  'mediumseagreen',
    'parked_or_stopped':  'mediumpurple',
    'unknown':            'lightgray',
}

def extract_physics_features(scenario):
    """
    Extracts per-frame physics features for each vehicle track:
      - speed:         1-second displacement magnitude (m/s)
      - acceleration:  change in speed over 1 second (m/s^2)
      - heading_change: absolute heading delta over 1 second (degrees)
      - is_sdc:        1 if this is the self-driving car, else 0
    
    Requires i >= 10 so we can compute acceleration from the previous
    1-second window (frame i-10 → i) vs. current window (frame i → i+10).
    """
    features = []
    sdc_index = scenario.sdc_track_index

    for track_id, track in enumerate(scenario.tracks):
        if track.object_type != scenario_pb2.Track.ObjectType.TYPE_VEHICLE:
            continue

        states = track.states
        is_sdc = 1 if track_id == sdc_index else 0

        # Need at least 21 frames: i-10 (prev), i (current), i+10 (future)
        for i in range(10, len(states) - 10):
            prev_state    = states[i - 10]
            current_state = states[i]
            future_state  = states[i + 10]

            if not (prev_state.valid and current_state.valid and future_state.valid):
                continue

            # Speed: displacement over the next 1 second
            dx_fwd = future_state.center_x - current_state.center_x
            dy_fwd = future_state.center_y - current_state.center_y
            speed_now = np.sqrt(dx_fwd**2 + dy_fwd**2)  # m/s (1s window)

            # Previous speed: displacement over the prior 1 second
            dx_prev = current_state.center_x - prev_state.center_x
            dy_prev = current_state.center_y - prev_state.center_y
            speed_prev = np.sqrt(dx_prev**2 + dy_prev**2)

            # Acceleration: change in speed over 1 second
            acceleration = speed_now - speed_prev  # m/s^2

            # Heading change: absolute delta over 1 second, normalized to [0, 180]
            raw_delta = future_state.heading - current_state.heading
            # Wrap to [-pi, pi] then convert to degrees
            raw_delta = (raw_delta + np.pi) % (2 * np.pi) - np.pi
            heading_change = abs(np.degrees(raw_delta))

            features.append({
                'track_id':      f"{scenario.scenario_id}+{track_id}",
                'is_sdc':        is_sdc,
                'speed':         speed_now,
                'acceleration':  acceleration,
                'heading_change': heading_change,
            })

    return pd.DataFrame(features)


def label_row(row):
    """
    Rule-based labeling against the README action definitions.
    Used AFTER clustering to interpret what each cluster represents —
    this is ground truth context, not a training signal.
    """
    s = row['speed']
    a = row['acceleration']
    h = row['heading_change']

    if s <= 1.0 and abs(a) <= 1.5:
        return 'parked_or_stopped'
    if h >= 45.0 and s < 10.0:
        return 'intersection_turn'
    if a < -8.0:
        return 'emergency_braking'
    if -8.0 <= a <= -1.0:
        return 'smooth_stop'
    if a >= 2.0 and s > 5.0:
        return 'aggressive_driving'
    if s > 1.0 and -1.0 <= a <= 2.0:
        return 'safe_cruising'
    return 'unknown'


def run_clustering(df, n_clusters=6):
    """
    Scales features and fits KMeans. Returns the fitted model,
    scaler, and the dataframe with cluster assignments.
    """
    feature_cols = ['speed', 'acceleration', 'heading_change']
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Fitting KMeans with {n_clusters} clusters on {len(df):,} data points...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Summarize each cluster by its mean physics values
    print("\n--- Cluster Centers (unscaled) ---")
    centers_unscaled = scaler.inverse_transform(kmeans.cluster_centers_)
    for i, center in enumerate(centers_unscaled):
        count = (df['cluster'] == i).sum()
        print(f"  Cluster {i}: speed={center[0]:.2f} m/s | "
              f"accel={center[1]:.2f} m/s² | "
              f"heading_change={center[2]:.2f}° | "
              f"n={count:,}")

    return kmeans, scaler, df, X_scaled


def plot_results(df, X_scaled, kmeans, n_clusters):
    """Generates all diagnostic plots into a single figure."""
    print("\nGenerating plots...")

    # Apply rule-based labels for reference
    df = df.copy()
    df['action_label'] = df.apply(label_row, axis=1)

    # 2D PCA projection for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]
    var_explained = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Vehicle Behavior Clustering — Waymo Open Motion Dataset", fontsize=16, fontweight='bold')

    cluster_colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    # Plot 1: PCA scatter colored by cluster 
    ax1 = fig.add_subplot(2, 3, 1)
    for c in range(n_clusters):
        mask = df['cluster'] == c
        ax1.scatter(df.loc[mask, 'pca_1'], df.loc[mask, 'pca_2'],
                    s=2, alpha=0.3, color=cluster_colors[c], label=f'Cluster {c}')
    ax1.set_title(f'PCA Projection by Cluster\n'
                  f'(PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%} variance)')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend(markerscale=4, fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: PCA scatter colored by rule-based action label 
    ax2 = fig.add_subplot(2, 3, 2)
    for action, color in ACTION_COLORS.items():
        mask = df['action_label'] == action
        if mask.any():
            ax2.scatter(df.loc[mask, 'pca_1'], df.loc[mask, 'pca_2'],
                        s=2, alpha=0.3, color=color, label=action.replace('_', ' ').title())
    ax2.set_title('PCA Projection by Rule-Based Action Label\n(for cluster interpretation)')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.legend(markerscale=4, fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Plot 3: Speed distribution per cluster 
    ax3 = fig.add_subplot(2, 3, 3)
    for c in range(n_clusters):
        speeds = df.loc[df['cluster'] == c, 'speed']
        ax3.hist(speeds, bins=40, alpha=0.5, color=cluster_colors[c],
                 label=f'Cluster {c}', density=True)
    ax3.set_title('Speed Distribution per Cluster')
    ax3.set_xlabel('Speed (m/s)')
    ax3.set_ylabel('Density')
    ax3.legend(fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Plot 4: Acceleration distribution per cluster 
    ax4 = fig.add_subplot(2, 3, 4)
    for c in range(n_clusters):
        accels = df.loc[df['cluster'] == c, 'acceleration']
        ax4.hist(accels, bins=40, alpha=0.5, color=cluster_colors[c],
                 label=f'Cluster {c}', density=True)
    ax4.set_title('Acceleration Distribution per Cluster')
    ax4.set_xlabel('Acceleration (m/s²)')
    ax4.set_ylabel('Density')
    ax4.legend(fontsize=8)
    ax4.grid(True, linestyle='--', alpha=0.5)

    # Plot 5: Speed vs Acceleration scatter (sampled for readability) 
    ax5 = fig.add_subplot(2, 3, 5)
    sample = df.sample(min(5000, len(df)), random_state=42)
    for c in range(n_clusters):
        mask = sample['cluster'] == c
        ax5.scatter(sample.loc[mask, 'speed'], sample.loc[mask, 'acceleration'],
                    s=8, alpha=0.4, color=cluster_colors[c], label=f'Cluster {c}')
    # Draw action boundary lines for reference
    ax5.axvline(x=5.0,  color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='Speed = 5 m/s')
    ax5.axhline(y=0.0,  color='gray',  linestyle='--', linewidth=0.8, alpha=0.5)
    ax5.axhline(y=-8.0, color='red',   linestyle='--', linewidth=0.8, alpha=0.5, label='Emergency brake = -8')
    ax5.set_title('Speed vs. Acceleration\n(sampled, cluster colored)')
    ax5.set_xlabel('Speed (m/s)')
    ax5.set_ylabel('Acceleration (m/s²)')
    ax5.legend(fontsize=7)
    ax5.grid(True, linestyle='--', alpha=0.5)

    # Plot 6: Cluster × Action label heatmap 
    ax6 = fig.add_subplot(2, 3, 6)
    action_order = list(ACTION_COLORS.keys())
    heatmap_data = pd.crosstab(df['cluster'], df['action_label'])
    # Ensure all action columns are present
    for col in action_order:
        if col not in heatmap_data.columns:
            heatmap_data[col] = 0
    heatmap_data = heatmap_data[action_order]
    # Normalize each row to percentages
    heatmap_pct = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

    im = ax6.imshow(heatmap_pct.values, cmap='YlOrRd', aspect='auto')
    ax6.set_xticks(range(len(action_order)))
    ax6.set_xticklabels([a.replace('_', '\n') for a in action_order], fontsize=7)
    ax6.set_yticks(range(n_clusters))
    ax6.set_yticklabels([f'Cluster {c}' for c in range(n_clusters)])
    ax6.set_title('Cluster Composition by Action Label\n(% of rows in each cluster)')
    plt.colorbar(im, ax=ax6, label='% of cluster')
    for i in range(n_clusters):
        for j in range(len(action_order)):
            val = heatmap_pct.values[i, j]
            if val > 5:
                ax6.text(j, i, f'{val:.0f}%', ha='center', va='center',
                         fontsize=7, color='black' if val < 60 else 'white')

    plt.tight_layout()
    plt.savefig("clustering_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("-> Saved clustering_results.png\n")

    # Also print a text summary of the dominant action per cluster
    print("--- Cluster Interpretation Summary ---")
    for c in range(n_clusters):
        cluster_rows = df[df['cluster'] == c]
        dominant = cluster_rows['action_label'].value_counts().idxmax()
        pct = cluster_rows['action_label'].value_counts().max() / len(cluster_rows) * 100
        sdc_pct = cluster_rows['is_sdc'].mean() * 100
        print(f"  Cluster {c}: dominant behavior = '{dominant}' ({pct:.1f}%) | "
              f"SDC share = {sdc_pct:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics-based vehicle behavior clustering on WOMD")
    parser.add_argument("--file_id",    type=str, default="00021", help="5-digit TFRecord file ID")
    parser.add_argument("--scenarios",  type=int, default=100,     help="Number of scenarios to process")
    parser.add_argument("--n_clusters", type=int, default=6,       help="Number of KMeans clusters")
    args = parser.parse_args()

    gcs_path = (f"gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/training/"
                f"training.tfrecord-{args.file_id}-of-01000")

    print(f"Loading dataset from: {gcs_path}")
    dataset = tf.data.TFRecordDataset(gcs_path, compression_type='')

    all_features = []
    print(f"Extracting physics features from {args.scenarios} scenarios...")
    for idx, data in enumerate(dataset.take(args.scenarios)):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(data.numpy())
        df = extract_physics_features(scenario)
        if not df.empty:
            all_features.append(df)
        print(f"  -> Scenario {idx+1}: {len(df):,} data points")

    final_df = pd.concat(all_features, ignore_index=True)
    print(f"\nTotal data points: {len(final_df):,}")
    print(f"SDC data points:   {final_df['is_sdc'].sum():,} "
          f"({final_df['is_sdc'].mean()*100:.1f}%)\n")

    # Print class distribution of rule-based labels (sanity check before clustering)
    final_df['action_label'] = final_df.apply(label_row, axis=1)
    print("--- Rule-Based Label Distribution (sanity check) ---")
    print(final_df['action_label'].value_counts().to_string())
    print()

    # Run clustering
    feature_cols = ['speed', 'acceleration', 'heading_change']
    kmeans, scaler, clustered_df, X_scaled = run_clustering(final_df, n_clusters=args.n_clusters)

    # Save clustered data
    clustered_df.to_csv("clustered_data.csv", index=False)
    print(f"\n-> Saved clustered_data.csv\n")

    # Generate all plots
    plot_results(clustered_df, X_scaled, kmeans, n_clusters=args.n_clusters)