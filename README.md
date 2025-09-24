# Deforestation Detection Baseline Models

This repository contains baseline models for deforestation detection using Sentinel-1 and Sentinel-2 satellite imagery.

## Dataset

The dataset contains:
- **Sentinel-1**: SAR imagery (2 bands) from July 22, 2020
- **Sentinel-2**: Optical imagery (4 bands) from July 27, 2020  
- **Masks**: Binary deforestation labels (1=deforestation, 2=non-deforestation)
- **Format**: 16 grid patches of 2816x2816 pixels each

## Models

### Deep Learning Models
1. **Sentinel2_Only**: Simple CNN using only Sentinel-2 optical data
2. **Sentinel1_Only**: Simple CNN using only Sentinel-1 SAR data
3. **Late_Fusion**: CNN with late fusion of Sentinel-1 and Sentinel-2
4. **UNet_Sentinel2**: U-Net architecture using Sentinel-2 data

### Traditional ML
5. **Random Forest**: Traditional ML baseline using statistical features

## Usage

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Train Deep Learning Models
```bash
python train.py
```

### Train Random Forest Baseline
```bash
python random_forest_baseline.py
```

## Configuration

Edit `config.py` to modify:
- Data paths
- Training parameters (batch size, learning rate, epochs)
- Model architecture parameters

## Results

The training script will:
- Split data into train/validation sets
- Train all models with progress tracking
- Save model checkpoints
- Generate performance plots
- Display final F1 scores for comparison

## File Structure

- `config.py`: Configuration parameters
- `data_loader.py`: PyTorch dataset and data loading utilities
- `models.py`: Model architectures (CNN, U-Net, Late Fusion)
- `train.py`: Main training script for deep learning models
- `random_forest_baseline.py`: Random Forest baseline implementation
- `requirements.txt`: Python dependencies
