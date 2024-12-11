import os
import random

def get_video_paths_by_subfolder(base_dir, extensions=[".mp4", ".avi", ".mov"]):
    """Retrieve video paths grouped by subfolder."""
    subfolder_videos = {}
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)
        if os.path.isdir(subfolder_path):
            videos = [
                os.path.relpath(os.path.join(subfolder_path, file), base_dir)
                for file in os.listdir(subfolder_path)
                if any(file.endswith(ext) for ext in extensions)
            ]
            subfolder_videos[subfolder] = videos
    return subfolder_videos

def split_data_per_subfolder(subfolder_videos, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split videos in each subfolder into train, val, and test sets."""
    train_split, val_split, test_split = [], [], []
    for subfolder, videos in subfolder_videos.items():
        random.shuffle(videos)
        total_videos = len(videos)
        train_end = int(total_videos * train_ratio)
        val_end = train_end + int(total_videos * val_ratio)
        train_split.extend(videos[:train_end])
        val_split.extend(videos[train_end:val_end])
        test_split.extend(videos[val_end:])
    return train_split, val_split, test_split

def save_splits_to_txt(train_videos, val_videos, test_videos, output_dir):
    """Save train, validation, and test splits to .txt files."""
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "train.txt")
    val_file = os.path.join(output_dir, "val.txt")
    test_file = os.path.join(output_dir, "test.txt")
    with open(train_file, "w") as f:
        f.write("\n".join(train_videos))
    with open(val_file, "w") as f:
        f.write("\n".join(val_videos))
    with open(test_file, "w") as f:
        f.write("\n".join(test_videos))
    print(f"Train, validation, and test splits saved to {output_dir}")

if __name__ == "__main__":
    # Path to the base directory containing all videos in subfolders
    base_dir = "./data"  # Update this path to point to your `data` directory
    output_dir = "./splits"  # Directory to save train, val, and test .txt files

    # Get video paths grouped by subfolder
    subfolder_videos = get_video_paths_by_subfolder(base_dir)
    print(f"Found videos in the following subfolders: {list(subfolder_videos.keys())}")

    # Split video paths into train, validation, and test sets
    train_videos, val_videos, test_videos = split_data_per_subfolder(subfolder_videos)

    # Save the splits to .txt files
    save_splits_to_txt(train_videos, val_videos, test_videos, output_dir)
