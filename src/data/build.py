import os
import numpy as np
from src.config import DATA_DIR

def build_dataset():
    classes = [name for name in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, name))]
    print(f"Detected classes: {classes}")

    target_filename = "preprocessed.npz"
    data_list, target_list = [], []
    
    for dirpath, _, filenames in os.walk(DATA_DIR):
        if target_filename in filenames:
            full_path = os.path.join(dirpath, target_filename)
            relative_dir_path = os.path.relpath(dirpath, DATA_DIR)
            class_label = os.path.normpath(relative_dir_path).split(os.sep)[0]
            
            data_single_file = np.load(full_path)["data"]
            ngestures = data_single_file.shape[0]
            target_single_file = np.array([classes.index(class_label)] * ngestures)
            
            data_list.append(data_single_file)
            target_list.append(target_single_file)
            print(f"Loaded {ngestures} gestures from {relative_dir_path}")

    # Concatenate all lists
    data = np.concatenate(data_list, axis=0)
    target = np.concatenate(target_list, axis=0)
    
    # Save the final dataset
    output_path = os.path.join(DATA_DIR, "alldata.npz")
    np.savez_compressed(output_path, data=data, target=target, labels=classes)
    print(f"Saved dataset to {output_path}. Shape: {data.shape}")

if __name__ == "__main__":
    build_dataset()