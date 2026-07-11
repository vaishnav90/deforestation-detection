# CanopyWatch — Deforestation AI Detector

Detect deforestation in Sentinel-1 (radar) and Sentinel-2 (optical) satellite imagery using trained CNN models, with a local web UI and REST API.

## Quick start

```bash
chmod +x run.sh
./run.sh
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Or manually:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

## What you can do

1. **Demo patches** — run detection on the 16 included cloud-free grid patches
2. **Upload GeoTIFFs** — analyze your own Sentinel-1 / Sentinel-2 rasters
3. **View results** — probability score, heatmap overlay, map view, downloadable GeoTIFF mask

## Models (presentation-ready)

| Model | Family | Calibrated F1 | Notes |
|-------|--------|---------------|-------|
| Simple CNN (Sentinel-2) ★ | Deep Learning | 1.00 | Best overall; corr 0.98 with true rate |
| Random Forest (S1+S2) | Classical ML | 1.00 | Fast statistical baseline |
| Logistic Regression (S1+S2) | Classical ML | 1.00 | Interpretable linear baseline |
| Late Fusion CNN (S1+S2) | Deep Learning | 0.97 | Multi-modal radar + optical |
| Simple CNN (Sentinel-1) | Deep Learning | 0.97 | Cloud-/night-proof radar |
| KNN (S1+S2) | Classical ML | 0.97 | Instance-based baseline |
| Robust Baseline CNN | Deep Learning | 0.93 | Residual CNN |

Use the **Compare models** tab to run all of these on one patch for slides.

## API

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/health` | Health check |
| `GET /api/v1/models` | List trained models |
| `GET /api/v1/leaderboard` | Presentation metrics table |
| `GET /api/v1/demo/patches` | List demo satellite patches |
| `POST /api/v1/predict/demo` | Run one model on a demo patch |
| `POST /api/v1/compare` | Side-by-side multi-model comparison |
| `POST /api/v1/predict` | Upload GeoTIFFs and run inference |

## Dataset

Place satellite data under `Claude AI Archive/` (see `DATA_SETUP.md`). Expected layout:

- Sentinel-1: `Claude AI Archive/1_CLOUD_FREE_DATASET/1_SENTINEL1/IMAGE_16_GRID/RASTER_{0-15}.tif`
- Sentinel-2: `Claude AI Archive/1_CLOUD_FREE_DATASET/2_SENTINEL2/IMAGE_16_GRID/RASTER_{0-15}.tif`
- Masks: `Claude AI Archive/3_TRAINING_MASKS/MASK_16_GRID/RASTER_{0-15}.tif`

## Project layout

- `models.py` — CNN architectures
- `data_loader.py` — training dataset
- `train_simple.py` — training script
- `inference.py` — model loading, tiling, prediction
- `api.py` — FastAPI backend
- `static/` — web UI
- `trained_models/` — saved weights

## Notes

Models are patch-level classifiers. Inference tiles large images, scores each tile, and assembles a spatial probability heatmap for visualization.
