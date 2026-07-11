import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
import torchvision.transforms as transforms
import config

class DeforestationDataset(Dataset):
    def __init__(self, patch_indices, transform=None, use_sentinel1=True, use_sentinel2=True):
        self.patch_indices = patch_indices
        self.transform = transform
        self.use_sentinel1 = use_sentinel1
        self.use_sentinel2 = use_sentinel2

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        patch_idx = self.patch_indices[idx]

        sentinel1_data = None
        sentinel2_data = None

        if self.use_sentinel1:
            sentinel1_path = f"{config.SENTINEL1_PATH}/RASTER_{patch_idx}.tif"
            with rasterio.open(sentinel1_path) as src:
                sentinel1_data = src.read()
                sentinel1_data = np.transpose(sentinel1_data, (1, 2, 0))

                sentinel1_data = (sentinel1_data - np.mean(sentinel1_data)) / (np.std(sentinel1_data) + 1e-8)

        if self.use_sentinel2:
            sentinel2_path = f"{config.SENTINEL2_PATH}/RASTER_{patch_idx}.tif"
            with rasterio.open(sentinel2_path) as src:
                sentinel2_data = src.read()
                sentinel2_data = np.transpose(sentinel2_data, (1, 2, 0))

                sentinel2_data = (sentinel2_data - np.mean(sentinel2_data)) / (np.std(sentinel2_data) + 1e-8)

        mask_path = f"{config.MASK_PATH}/RASTER_{patch_idx}.tif"
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        mask = (mask == 1).astype(np.float32)

        if self.use_sentinel1 and self.use_sentinel2:
            image = np.concatenate([sentinel1_data, sentinel2_data], axis=2)
        elif self.use_sentinel1:
            image = sentinel1_data
        elif self.use_sentinel2:
            image = sentinel2_data
        else:
            raise ValueError("At least one data source must be enabled")

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        if self.transform:
            image = self.transform(image)

        return image, mask
