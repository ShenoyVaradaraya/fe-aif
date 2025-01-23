import ast
import os

import cv2
import numpy as np
from scipy.special import softmax
from scipy.stats import norm


def load_trajectories_from_file(file_path):
    """
    Load trajectories from a text file in the current format.
    Expected format: Each line contains an object ID and its list of trails,
    with trails being lists of coordinates.
    
    Example format:
    0: deque([deque([(x1, y1), (x2, y2), ...], maxlen=30), ...], maxlen=30)
    1: deque([deque([(x1, y1), (x2, y2), ...], maxlen=30), ...], maxlen=30)
    
    Returns: Dictionary with object IDs as keys and numpy arrays of coordinates as values.
    """
    trajectories = {}

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                try:
                    # Split object ID and the raw data
                    object_id, raw_data = line.split(': ', 1)
                    object_id = int(object_id)  # Convert object ID to integer

                    # Replace deque and maxlen, then safely evaluate the structure
                    processed_data = raw_data.replace('deque', 'list').replace('maxlen=30', '')
                    trails = eval(processed_data)  # Convert to a nested list

                    # Flatten the trails into a single list of coordinates
                    all_coords = [coord for trail in trails for coord in trail]
                    trajectories[object_id] = np.array(all_coords)  # Convert to numpy array

                except Exception as e:
                    print(f"Warning: Skipping invalid line: {line}")
                    print(f"Error: {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return {}
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return {}

    return trajectories

def extract_features_from_coordinates(trajectory, fps=30, pixels_per_meter=50):
    if len(trajectory) < 3:  # Need at least 3 points for meaningful calculations
        return np.zeros(4)  # Return zero features for very short trajectories

    dt = 1.0 / fps
    trajectory_meters = trajectory / pixels_per_meter
    velocities = np.diff(trajectory_meters, axis=0) / dt
    speeds = np.linalg.norm(velocities, axis=1)

    # Aggregate features
    mean_speed = np.mean(speeds) if len(speeds) > 0 else 0
    stopping_tendency = np.mean(np.diff(speeds)) if len(speeds) > 1 else 0  # Speed reduction

    return np.array([mean_speed, stopping_tendency])  # Keep only speed-related features



class ActiveInferenceVehicle:
    def __init__(self):
        # Prior for stopping action (10% chance of stopping)
        self.prior_action_stop = 0.5
        self.prior_action_go = 0.5
        self.beliefs = np.array([self.prior_action_stop, self.prior_action_go])  # beliefs as array
        self.likelihood = self.define_likelihood()
        self.actions = ['stop', 'go']  # Example actions

    def define_likelihood(self):
        # Use speed to influence likelihood
        return lambda speed: norm.pdf(-speed, loc=0, scale=1)  # Inverse speed as stopping likelihood

    def scale_features(self, features):
        """
        Normalize the feature values to prevent issues with large feature values.
        """
        if len(features) > 0:
            mean = np.mean(features)
            std = np.std(features)
            return (features - mean) / std
            return features
    def update_beliefs(self, speed, learning_rate=0.1):
        likelihood_value = self.likelihood(speed)

        if likelihood_value <= 0:
            likelihood_value = 1e-10  # Prevent zero likelihood
        
        prediction_error = likelihood_value - self.beliefs[0]  # Only update stop belief
        self.beliefs[0] += learning_rate * prediction_error  # Update stop belief
        self.beliefs[1] = 1 - self.beliefs[0]  # Ensure beliefs sum to 1
        self.beliefs = np.clip(self.beliefs, 1e-10, 1 - 1e-10)  # Clip values to avoid invalid states

    def infer_action(self, features):
        # Calculate speed from features
        speed = features[0]  # Assuming the first feature is mean_speed
        self.update_beliefs(speed)
        return self.beliefs[0]  # Stop probability
    
# def process_trajectories_from_file(file_path, trajectory,model=None, chunk_size=3):
def process_trajectories_from_file(trajectory,model=None, chunk_size=3):
    """
    Load and process trajectories from a file in chunks of specified size (default: 3 coordinates).
    
    Parameters:
    - file_path: Path to the file containing trajectory data.
    - model: Optional model for inference (defaults to ActiveInferenceVehicle).
    - chunk_size: Number of coordinates to process at a time.

    Returns:
    - List of stop probabilities for each trajectory.
    """
    # Load trajectories
    # trajectories = load_trajectories_from_file(file_path)
    if not trajectory:
        print("No valid trajectories found in file.")
        return []
    
    # Initialize model if not provided
    # stop_probabilities = []
    if model is None:
        model = ActiveInferenceVehicle()

    # # Process each trajectory in chunks
    # for traj_id, trajectory in trajectories.items():
    #     print(f"Processing Trajectory ID: {traj_id}")
        # trajectory_stop_probs = []

        # Iterate through trajectory in chunks
    trajectory = np.array(list(trajectory))
    # print(f"Processing Trajectory: {trajectory}")
    for i in range(0, len(trajectory), chunk_size):
        chunk = trajectory[i:i + chunk_size]
        if len(chunk) < 3:  # Skip chunks smaller than 3 points
            continue
        
        # Extract features and infer stop probability
        features = extract_features_from_coordinates(np.array(chunk))
        stop_prob = model.infer_action(features)
        # trajectory_stop_probs.append(stop_prob)

        # print(f"Chunk {i // chunk_size} of Trajectory {traj_id}:")
        # print(f"    Stop Probability: {stop_prob:.3f}")
        # print(f"    Features: {features}")

    
    # Aggregate probabilities for the entire trajectory
    # avg_stop_prob = np.mean(trajectory_stop_probs) if trajectory_stop_probs else 0
    # stop_probabilities.append(avg_stop_prob)

        # print(f"Average Stop Probability for Trajectory {traj_id}: {avg_stop_prob:.3f}")
        # print()
    
    return (stop_prob if len(trajectory) > 3 else 0.0)

# Example usage
if __name__ == "__main__":
    file_path = "trails.txt"  # Update with your file path
    
    print(f"Processing trajectories from: {file_path}")
    stop_probabilities = process_trajectories_from_file(file_path)
    
    if stop_probabilities:
        print("\nSummary:")
        print(f"Processed {len(stop_probabilities)} trajectories")
        print(f"Average stop probability: {np.mean(stop_probabilities):.3f}")