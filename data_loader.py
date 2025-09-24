"""
DATA LOADER FOR DEFORESTATION DETECTION PROJECT
==============================================

This module provides a PyTorch Dataset class for loading satellite imagery
and deforestation masks. It supports both Sentinel-1 (radar) and Sentinel-2
(optical) data sources, with proper normalization and data augmentation.

Author: AI Assistant
Date: 2024
"""

import torch
from torch.utils.data import Dataset, DataLoader
import rasterio  # For reading geospatial raster data
import numpy as np
import torchvision.transforms as transforms
import config

class DeforestationDataset(Dataset):
    """
    PyTorch Dataset for deforestation detection using satellite imagery
    
    This dataset loads satellite imagery patches and their corresponding
    deforestation masks. It supports multiple data sources:
    - Sentinel-1: Radar data (VV, VH polarizations) - works in all weather
    - Sentinel-2: Optical data (RGB + NIR) - provides color information
    
    Features:
    - Automatic data normalization
    - Support for data augmentation
    - Flexible data source selection
    - Proper tensor formatting for PyTorch
    """
    
    def __init__(self, patch_indices, transform=None, use_sentinel1=True, use_sentinel2=True):
        """
        Initialize the deforestation dataset
        
        Args:
            patch_indices: List of patch indices to include in this dataset
            transform: Optional data augmentation transforms
            use_sentinel1: Whether to include Sentinel-1 radar data
            use_sentinel2: Whether to include Sentinel-2 optical data
        """
        self.patch_indices = patch_indices
        self.transform = transform
        self.use_sentinel1 = use_sentinel1
        self.use_sentinel2 = use_sentinel2
        
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.patch_indices)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            tuple: (image_tensor, mask_tensor)
                - image_tensor: Shape (channels, height, width)
                - mask_tensor: Shape (1, height, width) - binary deforestation mask
        """
        patch_idx = self.patch_indices[idx]
        
        sentinel1_data = None
        sentinel2_data = None
        
        # Load Sentinel-1 radar data if requested
        if self.use_sentinel1:
            sentinel1_path = f"{config.SENTINEL1_PATH}/RASTER_{patch_idx}.tif"
            with rasterio.open(sentinel1_path) as src:
                sentinel1_data = src.read()  # Shape: (bands, height, width)
                sentinel1_data = np.transpose(sentinel1_data, (1, 2, 0))  # Shape: (height, width, bands)
                
                # Normalize Sentinel-1 data to zero mean, unit variance
                # This is crucial for neural network training
                sentinel1_data = (sentinel1_data - np.mean(sentinel1_data)) / (np.std(sentinel1_data) + 1e-8)
        
        # Load Sentinel-2 optical data if requested
        if self.use_sentinel2:
            sentinel2_path = f"{config.SENTINEL2_PATH}/RASTER_{patch_idx}.tif"
            with rasterio.open(sentinel2_path) as src:
                sentinel2_data = src.read()  # Shape: (bands, height, width)
                sentinel2_data = np.transpose(sentinel2_data, (1, 2, 0))  # Shape: (height, width, bands)
                
                # Normalize Sentinel-2 data to zero mean, unit variance
                sentinel2_data = (sentinel2_data - np.mean(sentinel2_data)) / (np.std(sentinel2_data) + 1e-8)
        
        # Load deforestation mask
        mask_path = f"{config.MASK_PATH}/RASTER_{patch_idx}.tif"
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # Read first (and only) band
        
        # Convert mask to binary format
        # 1 = deforested pixel, 0 = forested pixel
        mask = (mask == 1).astype(np.float32)
        
        # Combine data sources based on configuration
        if self.use_sentinel1 and self.use_sentinel2:
            # Concatenate along channel dimension: (height, width, channels)
            image = np.concatenate([sentinel1_data, sentinel2_data], axis=2)
        elif self.use_sentinel1:
            image = sentinel1_data
        elif self.use_sentinel2:
            image = sentinel2_data
        else:
            raise ValueError("At least one data source must be enabled")
        
        # Convert to PyTorch tensors with proper format
        # PyTorch expects (channels, height, width) format
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # Add channel dimension
        
        # Apply data augmentation if specified
        if self.transform:
            image = self.transform(image)
        
        return image, mask

