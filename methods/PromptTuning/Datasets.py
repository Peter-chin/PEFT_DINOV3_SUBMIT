import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

def _load_rgb_image(img_path):
    with Image.open(img_path) as img:
        return img.convert("RGB").copy()

class ExcelDataset(Dataset):
    def __init__(self, dataframe, img_dir=None, transform=None, cache_images=False):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.df['full_path'] = self.df['image_id'].apply(
            lambda x: os.path.join(self.img_dir, x) if self.img_dir else x
        )

        self.df = self.df[self.df['full_path'].apply(os.path.exists)].reset_index(drop=True)
        self.transform = transform
        self.cache_images = cache_images
        self.cached_samples = None

        if self.cache_images:
            self.cached_samples = []
            for idx in range(len(self.df)):
                img_name = self.df.iloc[idx]['image_id']
                if self.img_dir is not None:
                    img_path = os.path.join(self.img_dir, img_name)
                else:
                    img_path = img_name

                label = self.df.iloc[idx]['tumor']
                # image = Image.open(img_path).convert("RGB")
                with Image.open(img_path) as img:
                    image = img.convert("RGB")

                if self.transform:
                    image = self.transform(image)

                self.cached_samples.append((image, torch.tensor(label, dtype=torch.long)))

    def __len__(self):
        return len(self.cached_samples) if self.cached_samples is not None else len(self.df)

    def __getitem__(self, idx):
        if self.cached_samples is not None:
            return self.cached_samples[idx]

        img_name = self.df.iloc[idx]['image_id']

        if self.img_dir is not None:
            img_path = os.path.join(self.img_dir, img_name)
        else:
            img_path = img_name

        label = self.df.iloc[idx]['tumor']

        # image = Image.open(img_path).convert("RGB")
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


class FracAtlasDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, cache_images=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.cache_images = cache_images
        self.cached_samples = None

        # Filter out corrupted images during initialization
        valid_indices = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            label = int(row["fractured"])
            img_name = row["image_id"]
            folder = "Fractured" if label == 1 else "Non_fractured"
            img_path = os.path.join(self.img_dir, folder, img_name)

            if not os.path.exists(img_path):
                print(f"[WARN] Image not found: {img_path}")
                continue

            try:
                # Try to open and load the image to catch truncated images
                with Image.open(img_path) as img:
                    img.load()  # This will raise OSError for truncated images
                valid_indices.append(idx)
            except (OSError, IOError):
                continue

        # Keep only valid rows
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"[INFO] Loaded {len(self.df)} valid images out of {len(df)} total")

        if self.cache_images:
            self.cached_samples = []
            for idx in range(len(self.df)):
                row = self.df.iloc[idx]
                label = int(row["fractured"])
                img_name = row["image_id"]
                folder = "Fractured" if label == 1 else "Non_fractured"
                img_path = os.path.join(self.img_dir, folder, img_name)

                # image = Image.open(img_path).convert("RGB")
                with Image.open(img_path) as img:
                    image = img.convert("RGB")

                if self.transform:
                    image = self.transform(image)

                self.cached_samples.append((image, torch.tensor(label, dtype=torch.long)))

    def __len__(self):
        return len(self.cached_samples) if self.cached_samples is not None else len(self.df)

    def __getitem__(self, idx):
        if self.cached_samples is not None:
            return self.cached_samples[idx]

        row = self.df.iloc[idx]
        label = int(row["fractured"])
        img_name = row["image_id"]
        folder = "Fractured" if label == 1 else "Non_fractured"
        img_path = os.path.join(self.img_dir, folder, img_name)

        # image = Image.open(img_path).convert("RGB")
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

MULTI_CLASS_LABELS = {
    "non tumor": 0,
    "osteochondroma": 1,
    "multiple osteochondromas": 2,
    "simple bone cyst": 3,
    "giant cell tumor": 4,
    "osteofibroma": 5,
    "synovial osteochondroma": 6,
    "other bt": 7,
    "osteosarcoma": 8,
    "other mt": 9
}

class BTXRDMultiClassDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        df = dataframe.reset_index(drop=True).copy()
        image_col = "image_id" if "image_id" in df.columns else "imageid"

        tumor_subtype_cols = [
            "osteochondroma",
            "multiple osteochondromas",
            "simple bone cyst",
            "giant cell tumor",
            "osteofibroma",
            "synovial osteochondroma",
            "other bt",
            "osteosarcoma",
            "other mt"
        ]

        kept_rows = []
        for _, row in df.iterrows():
            img_name = str(row[image_col]).strip()
            img_path = os.path.join(img_dir, img_name)

            if not os.path.exists(img_path):
                continue

            try:
                tumor = int(row["tumor"])
                positive_subtypes = [col for col in tumor_subtype_cols if int(row[col]) == 1]
            except Exception:
                continue

            if tumor == 0:
                if len(positive_subtypes) != 0:
                    continue
                label_name = "non tumor"
            else:
                if len(positive_subtypes) != 1:
                    continue
                label_name = positive_subtypes[0]

            row_dict = row.to_dict()
            row_dict["img_path"] = img_path
            row_dict["label_name"] = label_name
            row_dict["label"] = MULTI_CLASS_LABELS[label_name]
            kept_rows.append(row_dict)

        self.df = pd.DataFrame(kept_rows).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["img_path"]
        label = int(row["label"])

        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def preload_dataset_to_ram_multiclass(df, img_dir, datasets_type="BTXRD", transform=None):
    if datasets_type != "BTXRD":
        raise ValueError(f"Unsupported dataset type: {datasets_type}")

    dataset = BTXRDMultiClassDataset(df, img_dir=img_dir, transform=transform)
    filtered_df = dataset.df.reset_index(drop=True)
    return dataset, filtered_df


def multi_class_preprocess(df, datasets_type="BTXRD"):
    if datasets_type != "BTXRD":
        raise ValueError(f"Unsupported dataset type: {datasets_type}")

    labels = []
    tumor_subtype_cols = [
        "osteochondroma",
        "multiple osteochondromas",
        "simple bone cyst",
        "giant cell tumor",
        "osteofibroma",
        "synovial osteochondroma",
        "other bt",
        "osteosarcoma",
        "other mt"
    ]

    for _, row in df.reset_index(drop=True).iterrows():
        tumor = int(row["tumor"])
        positive_subtypes = [col for col in tumor_subtype_cols if int(row[col]) == 1]

        if tumor == 0:
            if len(positive_subtypes) != 0:
                labels.append(-1)
            else:
                labels.append(MULTI_CLASS_LABELS["non tumor"])
        else:
            if len(positive_subtypes) != 1:
                labels.append(-1)
            else:
                labels.append(MULTI_CLASS_LABELS[positive_subtypes[0]])

    return np.array(labels, dtype=np.int64)
