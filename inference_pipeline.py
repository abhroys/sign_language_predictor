import numpy as np
import json
import xgboost as xgb
from extract_keypoints import process_video
from train_xgb import flatten
import os
import pickle
import gc

# Paths
MODEL_PATH = r"D:\data_ISL\new_project\models\xgboost_xgboost.pickle.dat"
LABEL_MAP_PATH = r"D:\data_ISL\new_project\label_maps\label_map.json"
VIDEO_PATH = r"D:\data_ISL\new_project\data_all\chalk\a2_v2.mp4"

def load_label_map(label_map_path):
    """Load the label map from the given JSON file."""
    with open(label_map_path, "r") as file:
        return json.load(file)

def extract_keypoints_from_video(video_path, max_seq_len=200):
    """
    Process the video to extract keypoints and flatten them for inference.
    """
    # Extract raw keypoints from the video
    keypoints = process_video(video_path, save_dir=None)
    print(f"Keypoints extracted: {keypoints}")

    # Flatten the keypoints similar to training
    features = []
    for keypoint_type in ["pose", "hand1", "hand2"]:
        x = keypoints[f"{keypoint_type}_x"]
        y = keypoints[f"{keypoint_type}_y"]
        flattened_x = flatten(x, max_seq_len=max_seq_len)
        flattened_y = flatten(y, max_seq_len=max_seq_len)
        features.append(flattened_x)
        features.append(flattened_y)

    # Combine all flattened features
    return np.hstack(features)


def inference_pipeline(video_path, model_path, label_map_path):
    """
    Complete pipeline to predict the word represented in the given video.
    """
    # Load the trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load the label map
    label_map = load_label_map(label_map_path)
    reverse_label_map = {v: k for k, v in label_map.items()}

    # Extract and preprocess keypoints
    keypoints = process_video(video_path)
    if keypoints is None:
        raise ValueError(f"Failed to process video: {video_path}")

    # Flatten the keypoints for inference
    features = []
    for keypoint_type in ["pose", "hand1", "hand2"]:
        x = keypoints[f"{keypoint_type}_x"]
        y = keypoints[f"{keypoint_type}_y"]
        flattened_x = flatten(x, max_seq_len=200)
        flattened_y = flatten(y, max_seq_len=200)
        features.append(flattened_x)
        features.append(flattened_y)

    keypoints_flattened = np.hstack(features)

    if keypoints_flattened.ndim == 1:
        keypoints_flattened = keypoints_flattened.reshape(1, -1)

    # Predict using the model
    prediction = model.predict(keypoints_flattened)
    print("Raw Prediction:", prediction)

    predicted_label_index = np.argmax(prediction)

    # Map prediction to word
    predicted_word = reverse_label_map[predicted_label_index]
    return predicted_word
    




if __name__ == "__main__":
    print("Starting inference...")
    predicted_word = inference_pipeline(VIDEO_PATH, MODEL_PATH, LABEL_MAP_PATH)

