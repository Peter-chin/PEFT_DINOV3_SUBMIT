import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from Datasets import ExcelDataset, FracAtlasDataset
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms


class DataRegime:
    """Subsampling manager for data regimes (e.g. 5%, 10%, 50%, 100%).

    Attributes:
        x_data: ndarray or array-like, shape (n_samples, ...)
        y_data: ndarray, shape (n_samples,)
        data_regime: str (e.g. '10%', '50%', or 'all')
    """

    def __init__(self, x_data, y_data, data_regime: str):
        self.x_data = x_data
        self.y_data = np.asarray(y_data)
        self.data_regime = data_regime.strip().lower()
        
        # predefined regimes and their seeds for reproducibility
        self.regime_seeds = {'5%': 5, '10%': 10, '50%': 50, '100%': 42}
        self.available_regimes = ['5%', '10%', '50%', '100%']

    def _subsample(self, regime_str: str) -> Tuple:
        """Return (x_subset, y_subset) for the given regime percentage."""
        regime = regime_str.strip()
        
        # 100% or full: return everything
        if regime in ('100%', 'all'):
            return self.x_data, self.y_data

        # parse percentage
        try:
            if regime.endswith('%'):
                fraction = float(regime.rstrip('%')) / 100.0
            else:
                fraction = float(regime)
        except ValueError:
            raise ValueError(f"Invalid regime format: {regime}")

        if fraction <= 0 or fraction > 1:
            raise ValueError(f"Fraction must be in (0, 1], got {fraction}")

        seed = self.regime_seeds.get(regime, 42)

        # attempt stratified sampling
        try:
            x_tv, _, y_tv, _ = train_test_split(
                self.x_data, self.y_data,
                train_size=fraction,
                stratify=self.y_data,
                random_state=seed
            )
        except Exception:
            # fallback: random sampling without stratify
            n_samples = len(self.y_data)
            n_select = int(np.ceil(n_samples * fraction))
            rng = np.random.RandomState(seed)
            indices = rng.choice(n_samples, size=n_select, replace=False)
            x_tv = self.x_data[indices]
            y_tv = self.y_data[indices]

        return x_tv, y_tv

    def get_data(self):
        """Yield or return sampled data based on data_regime.

        - If data_regime is 'all': yield each regime's subset.
        - Otherwise: yield the single requested regime's subset.
        """
        if self.data_regime == 'all':
            
            for regime in self.available_regimes:
                x_sub, y_sub = self._subsample(regime)
                yield x_sub, y_sub, regime

        else:
            # yield the single requested regime
            x_sub, y_sub = self._subsample(self.data_regime)
            yield x_sub, y_sub, self.data_regime


if __name__ == "__main__":
    # TESTING
    data_path = "/vol/miltank/users/lechi/PEFT_DINOV3/datasets/FracAtlas/dataset.csv"
    img_dir = "/vol/miltank/users/lechi/PEFT_DINOV3/datasets/FracAtlas/images"
    
    kf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    
    df = pd.read_csv(data_path)

    x = df.index.values  # dummy x_data (indices)
    y = df['fractured'].values  # labels

    X_train_val, X_test, y_train_val, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=42)
    test_df = df.iloc[X_test].reset_index(drop=True)

    data_regime = 'all'  # or '5%', '10%', '50%', '100%'

    tfr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]), 
    ])

    regime_manager = DataRegime(X_train_val, y_train_val, data_regime)
    for x_sub, y_sub, regime in regime_manager.get_data():
        # five fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(x_sub, y_sub)):
            print(f"Fold: {fold}, Subset size: {len(x_sub)}, Regime: {regime}")
