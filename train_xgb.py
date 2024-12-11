import os
import glob
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import json
from models.Xgboost import Xgboost
from configs import XgbConfig
from utils import get_experiment_name, load_label_map
from tqdm.auto import tqdm


def flatten(arr, max_seq_len=200):
    arr = np.array(arr)
    if arr.shape[0] > max_seq_len:
        arr = arr[:max_seq_len, :]  # Truncate to max_seq_len
    arr = np.pad(arr, ((0, max_seq_len - arr.shape[0]), (0, 0)), "constant")
    arr = arr.flatten()
    return arr



def preprocess(df, label_map, mode):
    feature_cols = ["pose_x", "pose_y", "hand1_x", "hand1_y", "hand2_x", "hand2_y"]
    x, y = [], []
    pbar = tqdm(total=df.shape[0], desc=f"Processing {mode} file....")
    for i in range(df.shape[0]):
        row = df.loc[i, feature_cols]
        print(f"Row length before flattening: {[len(row[col]) for col in feature_cols]}")

        flatten_features = np.hstack(list(map(flatten, row.values)))

        x.append(flatten_features)
        y.append(label_map[df.loc[i, "label"]])
        pbar.update(1)
    x = np.stack(x)
    y = np.array(y)
    return x, y


def load_dataframe(files):
    dfs = []
    for file_path in files:
        with open(file_path, "r") as f:
            data = json.load(f)
            dfs.append(pd.DataFrame([data]))
    return pd.concat(dfs, axis=0).reset_index(drop=True)



def fit(args):
    # Update paths to match your dataset
    train_files = sorted(glob.glob(r"D:\data_ISL\new_project\data\train\*.json"))
    val_files = sorted(glob.glob(r"D:\data_ISL\new_project\data\val\*.json"))

    train_df = load_dataframe(train_files)
    print(train_df.head())
    print(train_df.columns)

    val_df = load_dataframe(val_files)

    label_map = load_label_map("label_maps/label_map.json")  # Update path as needed
    x_train, y_train = preprocess(train_df, label_map, "train")
    x_val, y_val = preprocess(val_df, label_map, "val")

    # Configure XGBoost for GPU
    config = XgbConfig(tree_method="gpu_hist")  # Enable GPU acceleration
    model = Xgboost(config=config)
    model.fit(x_train, y_train, x_val, y_val)

    # Save the model
    exp_name = get_experiment_name(args)
    save_path = os.path.join(args.save_dir, f"{exp_name}.pickle.dat")
    model.save(save_path)


def evaluate(args):
    # Update paths to match your dataset
    test_files = sorted(glob.glob(r"D:\data_ISL\new_project\data\test\*.json"))

    test_df = load_dataframe(test_files)

    label_map = load_label_map(r"D:\data_ISL\new_project\label_maps\label_map.json")  # Update path as needed
    x_test, y_test = preprocess(test_df, label_map, "test")

    # Load the model
    exp_name = get_experiment_name(args)
    config = XgbConfig(tree_method="gpu_hist")  # Ensure GPU usage
    model = Xgboost(config=config)
    load_path = os.path.join(args.save_dir, f"{exp_name}.pickle.dat")
    model.load(load_path)
    print("### Model loaded ###")

    # Evaluate the model
    test_preds = model(x_test)
    print("Test accuracy:", accuracy_score(y_test, test_preds))


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost model")
    parser.add_argument("--data_dir", type=str, default=r"D:\data_ISL\new_project\data", help="Path to dataset directory")
    parser.add_argument("--save_dir", type=str, default=r"D:\data_ISL\new_project\models", help="Path to save the model")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], required=True, help="Mode: train or evaluate")
    parser.add_argument("--model", type=str, default="xgboost", help="Model name for experiment tracking")  # Added argument
    args = parser.parse_args()

    if args.mode == "train":
        print("Starting training...")
        fit(args)
    elif args.mode == "evaluate":
        print("Starting evaluation...")
        evaluate(args)
