import os
from pathlib import Path
import shutil
from tqdm import tqdm

# ===============================
# FINAL COMBINED CLASSES
# ===============================
FINAL_CLASSES = [
    'bottle',        # 0
    'rock_hammer',   # 1
    'orange_hammer', # 2
    'aruco_tag'      # 3
]

# ===============================
# DATASET PATHS
# ===============================
DATASETS_ROOT = {
    "Data1": "C:\\Users\\Rayan Raad\\Desktop\\Training Files\\Data 1",
    "Data2": "C:\\Users\\Rayan Raad\\Desktop\\Training Files\\Data 2",
    "Data3": "C:\\Users\\Rayan Raad\\Desktop\\Training Files\\Data 3",
    # "Data4": "path_to_data4",  # skipping for now
    "Data5": "C:\\Users\\Rayan Raad\\Desktop\\Training Files\\Data 5"
}

# ===============================
# CLASS MAPPING FOR EACH DATASET
# Maps old class names -> FINAL_CLASSES index
# ===============================
DATASET_CLASS_MAPS = {
    "Data1": {
        'BOTTLE': 0,
        'HAMMER': 2,       # remap Hammer -> Orange Hammer
        'ROCK_HAMMER': 1   # Rock Hammer stays Rock Hammer
    },
    "Data2": {
        'bottle': 0,
        'mallet': 2        # Mallet -> Orange Hammer
    },
    "Data3": {
        'bottle': 0,
        'mallet': 2        # Mallet -> Orange Hammer
    },
    "Data5": {
        'ArUcoTag': 3,
        'Bottle': 0,
        'BrickHammer': 1,   # Brick Hammer -> Rock Hammer
        'OrangeHammer': 2
    }
}

# ===============================
# COMBINED DATASET OUTPUT
# ===============================
COMBINED_ROOT = Path("combined_dataset")

def create_combined_dirs():
    for split in ["train", "valid", "test"]:
        (COMBINED_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (COMBINED_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

# ===============================
# LABEL PROCESSING
# ===============================
def process_and_copy_labels(txt_file, dataset_map, split, dataset_name):
    new_lines = []
    with open(txt_file, "r") as f:
        lines = f.readlines()

    # Map numeric IDs in the file to final class IDs
    class_names = list(dataset_map.keys())
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        old_id = int(parts[0])
        if old_id >= len(class_names):
            continue
        old_class_name = class_names[old_id]
        if old_class_name not in dataset_map:
            continue
        new_id = dataset_map[old_class_name]
        if new_id is None:
            continue
        parts[0] = str(new_id)
        new_lines.append(" ".join(parts))

    if not new_lines:
        # skip empty labels
        return False

    # write to combined folder
    txt_name = txt_file.stem + ".txt"
    target_label_path = COMBINED_ROOT / split / "labels" / txt_name
    with open(target_label_path, "w") as f:
        f.write("\n".join(new_lines))

    # copy image
    for ext in [".jpg", ".png", ".jpeg"]:
        img_src = txt_file.parent.parent / "images" / (txt_file.stem + ext)
        if img_src.exists():
            img_dst = COMBINED_ROOT / split / "images" / (txt_file.stem + ext)
            shutil.copy2(img_src, img_dst)
            break  # found the image, stop
    return True

def process_dataset(dataset_name, dataset_path):
    print(f"\nProcessing {dataset_name}...")
    dataset_map = DATASET_CLASS_MAPS[dataset_name]

    for split in ["train", "valid", "test"]:
        label_dir = Path(dataset_path) / split / "labels"
        if not label_dir.exists():
            continue
        txt_files = list(label_dir.rglob("*.txt"))
        for txt_file in tqdm(txt_files, desc=f"{split} files", unit="file"):
            process_and_copy_labels(txt_file, dataset_map, split, dataset_name)

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    create_combined_dirs()
    for ds_name, ds_path in DATASETS_ROOT.items():
        process_dataset(ds_name, ds_path)

    # Write the combined YAML
    yaml_path = COMBINED_ROOT / "combined.yaml"
    yaml_content = f"""train: {COMBINED_ROOT / 'train' / 'images'}
val: {COMBINED_ROOT / 'valid' / 'images'}
test: {COMBINED_ROOT / 'test' / 'images'}

nc: {len(FINAL_CLASSES)}
names: {FINAL_CLASSES}
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print("\nAll datasets processed and merged into 'combined_dataset'.")
    print(f"YAML file created at: {yaml_path}")
