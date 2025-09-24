# Data Setup Instructions

## Satellite Data Requirements

This project requires satellite imagery data that is too large to include in the GitHub repository. You'll need to obtain the following data:

### Required Data Structure

```
Claude AI Archive/
├── 1_CLOUD_FREE_DATASET/
│   ├── 1_SENTINEL1/
│   │   └── IMAGE_16_GRID/
│   │       ├── RASTER_0.tif
│   │       ├── RASTER_1.tif
│   │       └── ... (RASTER_0.tif to RASTER_15.tif)
│   └── 2_SENTINEL2/
│       └── IMAGE_16_GRID/
│           ├── RASTER_0.tif
│           ├── RASTER_1.tif
│           └── ... (RASTER_0.tif to RASTER_15.tif)
├── 2_CLOUDY_DATASET/
│   ├── 1_SENTINEL1/
│   │   └── IMAGE_16_GRID/
│   │       └── ... (RASTER_0.tif to RASTER_15.tif)
│   └── 2_SENTINEL2/
│       └── IMAGE_16_GRID/
│           └── ... (RASTER_0.tif to RASTER_15.tif)
└── 3_TRAINING_MASKS/
    └── MASK_16_GRID/
        ├── RASTER_0.tif
        ├── RASTER_1.tif
        └── ... (RASTER_0.tif to RASTER_15.tif)
```

### Data Sources

- **Sentinel-1**: Radar data (VV, VH polarizations)
- **Sentinel-2**: Optical data (Red, Green, Blue, NIR bands)
- **Training Masks**: Binary deforestation masks

### Data Size

Total data size: ~12GB

### Alternative: Use Your Own Data

You can modify the code to work with your own satellite imagery by:

1. Updating the paths in `config.py`
2. Ensuring your data follows the same naming convention
3. Adjusting the number of patches in `config.py` if needed

### Data Download

Contact the project maintainer for data access or use your own satellite imagery dataset.
