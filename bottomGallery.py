import os
import shutil
import random
import pandas as pd

# ======= CONFIG – EDIT THIS =======
DATA_ROOT = r"E:\school\vscode_projects\outfitGenerator\kaggel\Deepfashion_dataset" 
LABELS_CSV = r"E:\school\vscode_projects\outfitGenerator\kaggel\Deepfashion_dataset\anno\anno\train_labels.csv"
IMG_ROOT   = r"E:\school\vscode_projects\outfitGenerator\kaggel\Deepfashion_dataset"

# How many example images to copy for the gallery per bottom type
MAX_PER_BOTTOM = 20

# Map from CSV category_name  ->  simple label used in the app & folder name
# TODO: change the LEFT side to match your train_labels.csv values
BOTTOM_CATEGORY_MAP = {
    "Jeans": "jeans",
    "Shorts": "shorts",
    "Skirt": "skirt",
    "Joggers": "joggers",
    "Dress": "dress_pants",  # or "dress" if you prefer
}

GALLERY_ROOT = "bottom_gallery"
# ===============================


def main():
    os.makedirs(GALLERY_ROOT, exist_ok=True)
    df = pd.read_csv(LABELS_CSV)

    # Keep only rows whose category_name is in our map
    df_bottoms = df[df["category_name"].isin(BOTTOM_CATEGORY_MAP.keys())]

    print("Found bottoms rows:", len(df_bottoms))

    for csv_cat, simple_label in BOTTOM_CATEGORY_MAP.items():
        subset = df_bottoms[df_bottoms["category_name"] == csv_cat]

        if subset.empty:
            print(f"[WARN] No rows found for bottom category '{csv_cat}'")
            continue

        n = min(MAX_PER_BOTTOM, len(subset))
        subset = subset.sample(n=n, random_state=42)

        out_dir = os.path.join(GALLERY_ROOT, simple_label)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Creating gallery for {csv_cat} -> {simple_label} ({n} images)")

        for _, row in subset.iterrows():
            rel_path = row["image_name"].replace("\\", "/")
            src_path = os.path.join(IMG_ROOT, rel_path)

            if not os.path.exists(src_path):
                continue

            dst_path = os.path.join(out_dir, os.path.basename(src_path))

            # Avoid overwriting if it already exists
            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)

    print("Done. Check the 'bottom_gallery' folder.")


if __name__ == "__main__":
    main()
