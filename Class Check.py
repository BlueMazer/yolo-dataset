import os
from collections import Counter

combined_folder = "combined_dataset/train/labels"  # path to your labels folder

counter = Counter()

# Walk through all subfolders (if any) and count class IDs
for root, dirs, files in os.walk(combined_folder):
    for file in files:
        if file.endswith(".txt"):
            lbl_path = os.path.join(root, file)
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls = int(parts[0])
                    counter[cls] += 1

print("Class counts in train set:")
for cls_id, count in counter.items():
    print(f"Class {cls_id}: {count} instances")
