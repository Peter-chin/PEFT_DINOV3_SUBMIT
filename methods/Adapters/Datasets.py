import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ExcelDataset(Dataset):
    def __init__(self, dataframe, img_dir=None, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.df['full_path'] = self.df['image_id'].apply(
          lambda x: os.path.join(self.img_dir, x) if self.img_dir else x
        )

        self.df = self.df[self.df['full_path'].apply(os.path.exists)].reset_index(drop=True)
        self.transform = transform 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

          # ======================
        img_name = self.df.iloc[idx]['image_id']

        if self.img_dir is not None:
          img_path = os.path.join(self.img_dir, img_name)
        else:
          img_path = img_name
                  
        # ======================
        label = self.df.iloc[idx]['tumor']

        # ======================
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

class FracAtlasDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        
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
            except (OSError, IOError) as e:
                # print(f"[WARN] Corrupted/truncated image skipped: {img_path} | Error: {str(e)}")
                continue
        
        # Keep only valid rows
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"[INFO] Loaded {len(self.df)} valid images out of {len(df)} total")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row["fractured"])
        img_name = row["image_id"]
        folder = "Fractured" if label == 1 else "Non_fractured"
        img_path = os.path.join(self.img_dir, folder, img_name)
        
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)
