import numpy as np
import pandas as pd 
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def extract_features(scenario):
    """Extracts X, Y, and 1 second magntiude displacement for vehicles"""
    features = []
    
    # identify bike lane coordinates for labeling
    bike_lanes = []
    for feature in scenario.map_features:
        if feature.HasField('lane') and feature.lane.type == 3: # bike
            bike_lanes.extend((p.x, p.y) for p in feature.lane.polyline)

    # bounding box to check for bike lane
    def in_bike_lane(x, y):
        if not bike_lanes: return 0
        distances = [np.sqrt((x - bx)**2 + (y - by)**2) for bx, by in bike_lanes]
        return 1 if min(distances) < 2.0 else 0 # 2 meters threshold

    # Iterates through tracks
    for track_id, track in enumerate(scenario.tracks):
        if track.object_type != scenario_pb2.Track.ObjectType.TYPE_VEHICLE:
            continue
        states = track.states
        # tracking 1 second of data (10 frames at 10Hz)
        for i in range(len(states) - 10):
            current_state = states[i]
            future_state = states[i+10] # 1 sec later
            if current_state.valid and future_state.valid:
                dx = future_state.center_x - current_state.center_x
                dy = future_state.center_y - current_state.center_y
                magnitude = np.sqrt(dx**2 + dy**2)
                features.append({
                    'track_id': f"{scenario.scenario_id}+{track_id}", # unique ID for grouping
                    'x': current_state.center_x,
                    'y': current_state.center_y,
                    'displacement': magnitude,
                    'is_bike_lane': in_bike_lane(current_state.center_x, current_state.center_y)
                    })
    return pd.DataFrame(features)

def create_splits(df):
    """Splitting in 60/20/20 splits"""
    # 60% train
    gss = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df['track_id']))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    # 2nd split: 20% val, 20% test
    gss_temp = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(gss.split(temp_df, groups=temp_df['track_id']))
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Point this to the TFRecord file
    gcs_path = "gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/training/training.tfrecord-00021-of-01000"

    print(f"Loading dataset from: {gcs_path}")
    dataset = tf.data.TFRecordDataset(gcs_path, compression_type='')

    all_features = []

    # Process the first x scenarios in the record 
    print("Extracting features from scenarios...")
    for idx, data in enumerate(dataset.take(100)):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(data.numpy())
        df = extract_features(scenario)
        all_features.append(df)
        print(f"  -> Processed Scenario {idx+1}: Extracted {len(df)} 1-second data points.")

    # Combine all scenarios into one master dataframe
    final_df = pd.concat(all_features, ignore_index=True)
    print(f"\nTotal data points extracted: {len(final_df)}")
    print("Splitting data ensuring to ensure vehicles are not mixed.")
    train, val, test = create_splits(final_df)

    # Saving into CSV files directory within current directory
    train.to_csv("train_data.csv", index=False)
    val.to_csv("val_data.csv", index=False)
    test.to_csv("test_data.csv", index=False)

    print("Finished and saved into directory")
