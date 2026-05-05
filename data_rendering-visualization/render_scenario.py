import argparse
import os
import tensorflow as tf 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from waymo_open_dataset.protos import scenario_pb2

# Suppress TF logging clutter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_scenario(tfrecord_path):
    print(f"Loading dataset from: {tfrecord_path}")
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    for data in dataset.take(1):
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(data.numpy())
        return scenario
    return None

def extract_map_features(scenario):
    map_lanes, map_bike_lanes, map_road_lines, map_road_edges = [], [], [], []
    for feature in scenario.map_features:
        if feature.HasField('lane'):
            coords = ([p.x for p in feature.lane.polyline], [p.y for p in feature.lane.polyline])
            if feature.lane.type == 3:
                map_bike_lanes.append(coords)
            else:
                map_lanes.append(coords)
        elif feature.HasField('road_line'):
            coords = ([p.x for p in feature.road_line.polyline], [p.y for p in feature.road_line.polyline])
            map_road_lines.append(coords)
        elif feature.HasField('road_edge'):
            coords = ([p.x for p in feature.road_edge.polyline], [p.y for p in feature.road_edge.polyline])
            map_road_edges.append(coords)
    return map_lanes, map_bike_lanes, map_road_lines, map_road_edges

def render_static_plot(scenario, out_file="scenario_plot.png"):
    fig, ax = plt.subplots(figsize=(10, 10))
    map_lanes, map_bike_lanes, map_road_lines, map_road_edges = extract_map_features(scenario)
    
    # Draw Map
    for x, y in map_lanes: ax.plot(x, y, color='lightgray', linestyle='--', linewidth=1, alpha=0.6, zorder=1)
    for x, y in map_bike_lanes: ax.plot(x, y, color='lightgreen', linestyle='--', linewidth=1, alpha=0.6, zorder=1)
    for x, y in map_road_lines: ax.plot(x, y, color='darkgray', linestyle='-', linewidth=1.5, zorder=1)
    for x, y in map_road_edges: ax.plot(x, y, color='black', linewidth=2, zorder=2)

    # Draw Full Agent Trajectories
    for i, track in enumerate(scenario.tracks):
        x = [state.center_x for state in track.states if state.valid]
        y = [state.center_y for state in track.states if state.valid]
        if len(x) > 1:
            color = 'red' if i == scenario.sdc_track_index else 'blue'
            z = 20 if i == scenario.sdc_track_index else 10
            ax.plot(x, y, color=color, alpha=0.5, zorder=z)
            ax.arrow(x[0], y[0], x[-1]-x[0], y[-1]-y[0], color=color, alpha=0.8, head_width=2.5, zorder=15)

    ax.set_title(f"Scenario ID: {scenario.scenario_id}")
    plt.grid(True)
    plt.savefig(out_file)
    print(f"Static plot saved to {out_file}")

def render_video(scenario, out_file="scenario_video.mp4"):
    fig, ax = plt.subplots(figsize=(10, 10))
    map_lanes, map_bike_lanes, map_road_lines, map_road_edges = extract_map_features(scenario)
    
    agent_data, all_x, all_y = [], [], []
    for i, track in enumerate(scenario.tracks):
        states = [s for s in track.states]
        if states:
            agent_data.append({'type': track.object_type, 'is_sdc': i == scenario.sdc_track_index, 'states': states})
            for s in states:
                if s.valid:
                    all_x.append(s.center_x); all_y.append(s.center_y)

    x_min, x_max = min(all_x) - 10, max(all_x) + 10
    y_min, y_max = min(all_y) - 10, max(all_y) + 10

    def update(frame_index):
        ax.clear()
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_title(f"Scenario: {scenario.scenario_id} | Step: {frame_index}")
        
        for x, y in map_lanes: ax.plot(x, y, color='lightgray', linestyle='--', linewidth=1, alpha=0.6, zorder=1)
        for x, y in map_bike_lanes: ax.plot(x, y, color='lightgreen', linestyle='--', linewidth=1, alpha=0.6, zorder=1)
        for x, y in map_road_lines: ax.plot(x, y, color='darkgray', linestyle='-', linewidth=1.5, zorder=1)
        for x, y in map_road_edges: ax.plot(x, y, color='black', linewidth=2, zorder=2)

        for agent in agent_data:
            state = agent['states'][frame_index]
            if state.valid:
                color = 'red' if agent['is_sdc'] else 'blue'
                ax.scatter(state.center_x, state.center_y, color=color, s=50, zorder=10)

    ani = FuncAnimation(fig, update, frames=len(scenario.timestamps_seconds), interval=100)
    print(f"Rendering video to {out_file} (this may take a moment)...")
    ani.save(out_file, writer='ffmpeg', fps=10)
    print("Render complete.")

if __name__ == "__main__":
    file_id = "00021"
    parser = argparse.ArgumentParser(description="Render WOMD Scenarios")
    parser.add_argument("--file_id", type=str, default= file_id, help="The 5 digit ID for the TF record")
    parser.add_argument("--mode", type=str, choices=['static', 'video'], required=True, help="Render a 'static' graph or a 'video' MP4")
    
    args = parser.parse_args()
    gcs_path = f"gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/training/training.tfrecord-{args.file_id}-of-01000"
    scenario_data = load_scenario(gcs_path)
    
    if args.mode == 'static':
        render_static_plot(scenario_data)
    elif args.mode == 'video':
        render_video(scenario_data)
