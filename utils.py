import random
import os
import numpy as np
import json


def seed_everything(seed):
    """
    Set seed for reproducibility across random, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_json(path):
    """
    Load a JSON file from the given path.
    """
    with open(path, "r") as f:
        json_file = json.load(f)
    return json_file


def load_label_map(label_map_path):
    """
    Load label map JSON from a specified path.
    """
    return load_json(label_map_path)


def get_experiment_name(args):
    """
    Generate a name for the experiment based on arguments.
    """
    exp_name = "xgboost_"  # Hardcoded since only XGBoost is used
    exp_name += args.model
    return exp_name


class AverageMeter:
    """
    Track and compute average values (e.g., loss, accuracy) during training.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Implements early stopping to terminate training when validation score stops improving.
    """
    def __init__(self, patience=5, mode="min", delta=0.0):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, model_path, epoch_score, model):
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Saving model!".format(
                    self.val_score, epoch_score
                )
            )
            model.save(model_path)  # Save XGBoost model directly
        self.val_score = epoch_score
