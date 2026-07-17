import glob
import json
import os

import numpy as np
import torch
import monai.transforms as mt
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _BaseSegDataset(Dataset):
    def __init__(self, img_dir, img_size=448, augment=False):
        self.img_dir = img_dir
        self.img_size = img_size
        self.augment = augment
        self.records = []

        self.aug_transform = mt.Compose(
            [
                mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
                mt.RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
                mt.RandRotated(
                    keys=["image", "mask"],
                    range_x=0.17,
                    prob=0.5,
                    keep_size=True,
                    mode=("bilinear", "nearest"),
                    padding_mode="zeros",
                ),
                mt.RandZoomd(
                    keys=["image", "mask"],
                    prob=0.5,
                    min_zoom=0.9,
                    max_zoom=1.1,
                    keep_size=True,
                    mode=("bilinear", "nearest"),
                ),
                mt.RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.8, 1.2)),
                mt.RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
            ]
        )

    def _resolve_path(self, file_name):
        candidates = [
            os.path.join(self.img_dir, file_name),
            os.path.join(self.img_dir, "Fractured", file_name),
            os.path.join(self.img_dir, "Non_fractured", file_name),
            os.path.join(self.img_dir, "images", file_name),
            os.path.join(self.img_dir, "images", "Fractured", file_name),
            os.path.join(self.img_dir, "images", "Non_fractured", file_name),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _validate_records(self, records, source):
        usable, missing, corrupt = [], 0, 0
        for rec in records:
            path = rec.get("path")
            if path is None or not os.path.exists(path):
                missing += 1
                continue
            try:
                with Image.open(path) as img:
                    img.load()
            except (OSError, IOError):
                corrupt += 1
                print(f"[WARN] Skipping corrupt image: {path}")
                continue
            usable.append(rec)
        print(
            f"[INFO] {type(self).__name__}: {len(usable)} usable images "
            f"({missing} missing, {corrupt} corrupt/unreadable) from {source}"
        )
        return usable

    def _rasterize_mask(self, rec, width, height):
        raise NotImplementedError

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        image = None
        for attempt in range(3):
            try:
                image = Image.open(rec["path"]).convert("RGB")
                break
            except (OSError, IOError):
                if attempt == 2:
                    print(f"[WARN] Unreadable image (returning blank): {rec['path']}")
                    image = Image.new("RGB", (self.img_size, self.img_size), 0)

        width, height = image.size
        mask = self._rasterize_mask(rec, width, height)

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        image = np.asarray(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()

        mask = torch.from_numpy(np.asarray(mask, dtype=np.float32)).unsqueeze(0)

        if self.augment:
            data = self.aug_transform({"image": image, "mask": mask})
            image = data["image"].float()
            mask = data["mask"].float()
            image = torch.clamp(image, 0.0, 1.0)
            mask = (mask > 0.5).float()

        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        image = (image - mean) / std

        return image, mask


class FracAtlasCocoSegDataset(_BaseSegDataset):
    def __init__(self, coco_json, img_dir, img_size=448, augment=False):
        super().__init__(img_dir, img_size, augment=augment)

        with open(coco_json, "r") as f:
            coco = json.load(f)

        images = {img["id"]: img for img in coco["images"]}

        polygons_by_image = {img_id: [] for img_id in images}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            seg = ann.get("segmentation", [])
            if isinstance(seg, list):
                for poly in seg:
                    if len(poly) >= 6:
                        polygons_by_image[img_id].append(poly)

        records = []
        for img_id, meta in images.items():
            records.append(
                {
                    "path": self._resolve_path(meta["file_name"]),
                    "polygons": polygons_by_image.get(img_id, []),
                }
            )

        self.records = self._validate_records(records, coco_json)

    def _rasterize_mask(self, rec, width, height):
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for poly in rec["polygons"]:
            xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
            draw.polygon(xy, outline=1, fill=1)
        return mask


class BTXRDSegDataset(_BaseSegDataset):
    def __init__(self, ann_dir, img_dir, img_size=448, augment=False):
        super().__init__(img_dir, img_size, augment=augment)

        records = []
        for ann_path in sorted(glob.glob(os.path.join(ann_dir, "*.json"))):
            try:
                with open(ann_path, "r") as f:
                    ann = json.load(f)
            except (OSError, IOError, json.JSONDecodeError):
                print(f"[WARN] Skipping unreadable annotation: {ann_path}")
                continue

            file_name = ann.get("imagePath") or (
                os.path.splitext(os.path.basename(ann_path))[0] + ".jpeg"
            )
            records.append(
                {
                    "path": self._resolve_path(os.path.basename(file_name)),
                    "shapes": ann.get("shapes", []),
                }
            )

        self.records = self._validate_records(records, ann_dir)

    def _rasterize_mask(self, rec, width, height):
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for shape in rec["shapes"]:
            pts = shape.get("points", [])
            shape_type = shape.get("shape_type", "polygon")
            if shape_type == "rectangle" and len(pts) >= 2:
                (x0, y0), (x1, y1) = pts[0], pts[1]
                draw.rectangle(
                    [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                    outline=1,
                    fill=1,
                )
            elif len(pts) >= 3:
                draw.polygon([(float(x), float(y)) for x, y in pts], outline=1, fill=1)
        return mask


def build_seg_dataset(dataset_type, data_root, img_size=448, augment=False):
    dtype = dataset_type.strip().lower()
    img_dir = os.path.join(data_root, "images")

    if dtype == "fracatlas":
        coco_json = os.path.join(
            data_root, "Annotations", "COCO JSON", "COCO_fracture_masks.json"
        )
        return FracAtlasCocoSegDataset(coco_json, img_dir, img_size=img_size, augment=augment)
    elif dtype == "btxrd":
        ann_dir = os.path.join(data_root, "Annotations")
        return BTXRDSegDataset(ann_dir, img_dir, img_size=img_size, augment=augment)
    else:
        raise ValueError(
            f"Unsupported dataset_type {dataset_type!r}; expected 'FracAtlas' or 'BTXRD'"
        )
