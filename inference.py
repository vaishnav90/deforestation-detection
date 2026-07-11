from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.transform import Affine

import config
from models import LateFusionCNN, RobustBaselineCNN, SimpleCNN
from simple_ml_baselines import extract_features_from_patch

MODELS_DIR = "trained_models"
OUTPUT_DIR = "outputs"
CALIBRATION_PATH = os.path.join(MODELS_DIR, "model_calibration.json")
DEFAULT_TILE_SIZE = 704
DEFAULT_THRESHOLD = 0.5

DL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "simplecnn_sentinel2": {
        "display_name": "Simple CNN (Sentinel-2 optical)",
        "architecture": "SimpleCNN",
        "family": "deep_learning",
        "weights": "simplecnn_sentinel2_best.pth",
        "results": "simplecnn_sentinel2_results.json",
        "use_sentinel1": False,
        "use_sentinel2": True,
        "input_channels": 4,
        "recommended": True,
        "blurb": "Best overall — strongest correlation with true deforestation rate.",
    },
    "latefusion_combined": {
        "display_name": "Late Fusion CNN (S1 + S2)",
        "architecture": "LateFusionCNN",
        "family": "deep_learning",
        "weights": "latefusion_combined_best.pth",
        "results": "latefusion_combined_results.json",
        "use_sentinel1": True,
        "use_sentinel2": True,
        "input_channels": 6,
        "recommended": False,
        "blurb": "Multi-modal fusion of radar + optical. Strong ranking after calibration.",
    },
    "simplecnn_sentinel1": {
        "display_name": "Simple CNN (Sentinel-1 radar)",
        "architecture": "SimpleCNN",
        "family": "deep_learning",
        "weights": "simplecnn_sentinel1_best.pth",
        "results": "simplecnn_sentinel1_results.json",
        "use_sentinel1": True,
        "use_sentinel2": False,
        "input_channels": 2,
        "recommended": False,
        "blurb": "Works in clouds / night — radar-only detection.",
    },
    "robustbaseline_combined": {
        "display_name": "Robust Baseline CNN (S1 + S2)",
        "architecture": "RobustBaselineCNN",
        "family": "deep_learning",
        "weights": "robustbaseline_combined_best.pth",
        "results": "robustbaseline_combined_results.json",
        "use_sentinel1": True,
        "use_sentinel2": True,
        "input_channels": 6,
        "recommended": False,
        "blurb": "Residual CNN baseline with strong regularization.",
    },
}

ML_REGISTRY: Dict[str, Dict[str, Any]] = {
    "randomforest_combined": {
        "display_name": "Random Forest (S1 + S2 stats)",
        "architecture": "RandomForest",
        "family": "classical_ml",
        "weights": "randomforest_combined_sklearn.pkl",
        "results": "randomforest_combined_sklearn_results.json",
        "use_sentinel1": True,
        "use_sentinel2": True,
        "input_channels": 6,
        "recommended": False,
        "blurb": "Classical ensemble on band statistics — fast and interpretable.",
    },
    "logisticregression_combined": {
        "display_name": "Logistic Regression (S1 + S2 stats)",
        "architecture": "LogisticRegression",
        "family": "classical_ml",
        "weights": "logisticregression_combined_sklearn.pkl",
        "results": "logisticregression_combined_sklearn_results.json",
        "use_sentinel1": True,
        "use_sentinel2": True,
        "input_channels": 6,
        "recommended": False,
        "blurb": "Linear baseline — good for explaining feature effects.",
    },
    "knn_combined": {
        "display_name": "K-Nearest Neighbors (S1 + S2 stats)",
        "architecture": "KNN",
        "family": "classical_ml",
        "weights": "knn_combined_sklearn.pkl",
        "results": "knn_combined_sklearn_results.json",
        "use_sentinel1": True,
        "use_sentinel2": True,
        "input_channels": 6,
        "recommended": False,
        "blurb": "Instance-based baseline for comparison slides.",
    },
    "randomforest_sentinel2": {
        "display_name": "Random Forest (Sentinel-2 stats)",
        "architecture": "RandomForest",
        "family": "classical_ml",
        "weights": "randomforest_sentinel2_sklearn.pkl",
        "results": "randomforest_sentinel2_sklearn_results.json",
        "use_sentinel1": False,
        "use_sentinel2": True,
        "input_channels": 4,
        "recommended": False,
        "blurb": "Optical-only classical ML baseline.",
    },
}

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {**DL_REGISTRY, **ML_REGISTRY}

@dataclass
class PredictionResult:
    job_id: str
    model_id: str
    probability: float
    is_deforested: bool
    threshold: float
    deforestation_fraction: float
    mean_confidence: float
    tile_count: int
    image_size: Tuple[int, int]
    mask_path: str
    heatmap_path: str
    preview_path: Optional[str]
    bounds: Optional[List[float]]
    crs: Optional[str]
    estimated_rate: float = 0.0
    model_family: str = "deep_learning"
    calibrated: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)

def ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR

def load_calibration() -> Dict[str, Any]:
    if not os.path.exists(CALIBRATION_PATH):
        return {}
    with open(CALIBRATION_PATH) as f:
        return json.load(f)

def get_model_threshold(model_id: str, override: Optional[float] = None) -> Tuple[float, bool]:
    if override is not None:
        return float(override), False
    calib = load_calibration().get(model_id)
    if calib and "optimal_threshold" in calib:
        return float(calib["optimal_threshold"]), True
    return DEFAULT_THRESHOLD, False

def estimate_rate(model_id: str, raw_score: float) -> float:
    calib = load_calibration().get(model_id)
    if not calib:
        return float(np.clip(raw_score, 0, 1))
    slope = calib.get("calibration_slope", 1.0)
    intercept = calib.get("calibration_intercept", 0.0)
    return float(np.clip(slope * raw_score + intercept, 0, 1))

def list_available_models() -> List[Dict[str, Any]]:
    calib = load_calibration()
    models = []
    for model_id, meta in MODEL_REGISTRY.items():
        weights_path = os.path.join(MODELS_DIR, meta["weights"])
        available = os.path.exists(weights_path)
        entry = {
            "id": model_id,
            "name": meta["display_name"],
            "architecture": meta["architecture"],
            "family": meta["family"],
            "use_sentinel1": meta["use_sentinel1"],
            "use_sentinel2": meta["use_sentinel2"],
            "input_channels": meta["input_channels"],
            "recommended": meta["recommended"],
            "blurb": meta.get("blurb", ""),
            "available": available,
        }
        results_path = os.path.join(MODELS_DIR, meta["results"])
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)
            entry["best_f1"] = results.get("best_f1") or results.get("val", {}).get("f1")
            entry["best_epoch"] = results.get("best_epoch")
        if model_id in calib:
            c = calib[model_id]

            entry["val_f1"] = c.get("val_f1")
            entry["val_accuracy"] = c.get("val_accuracy")
            entry["val_n"] = c.get("val_n", 6)
            entry["corr_with_rate"] = c.get("val_corr") or c.get("all_corr")
            entry["all_corr"] = c.get("all_corr")
            entry["loo_f1"] = c.get("loo_f1")
            entry["optimal_threshold"] = c.get("optimal_threshold")
            entry["val_confusion"] = c.get("val_confusion")

            entry["calibrated_f1"] = entry["val_f1"]
            entry["calibrated_accuracy"] = entry["val_accuracy"]
        models.append(entry)

    def sort_key(m: Dict[str, Any]) -> Tuple:
        corr = m.get("corr_with_rate") or 0
        return (0 if m.get("recommended") else 1, -float(corr))

    models.sort(key=sort_key)
    return models

def list_demo_patches() -> List[Dict[str, Any]]:
    patches = []
    for idx in range(config.NUM_PATCHES):
        s1 = os.path.join(config.SENTINEL1_PATH, f"RASTER_{idx}.tif")
        s2 = os.path.join(config.SENTINEL2_PATH, f"RASTER_{idx}.tif")
        mask = os.path.join(config.MASK_PATH, f"RASTER_{idx}.tif")
        if not (os.path.exists(s1) and os.path.exists(s2)):
            continue
        ground_truth = None
        if os.path.exists(mask):
            with rasterio.open(mask) as src:
                m = (src.read(1) == 1).astype(np.float32)
                ground_truth = float(np.mean(m))
        patches.append(
            {
                "id": idx,
                "label": f"Patch {idx}",
                "has_sentinel1": os.path.exists(s1),
                "has_sentinel2": os.path.exists(s2),
                "has_mask": os.path.exists(mask),
                "ground_truth_deforestation_rate": ground_truth,
            }
        )
    return patches

def get_leaderboard() -> Dict[str, Any]:
    models = list_available_models()

    majority_f1 = 10 / 11
    majority_acc = 5 / 6

    rows = []
    for m in models:
        if not m["available"]:
            continue
        val_f1 = m.get("val_f1") or m.get("calibrated_f1")
        beats_majority = None
        if val_f1 is not None:
            beats_majority = float(val_f1) > majority_f1 + 0.01
        rows.append(
            {
                "id": m["id"],
                "name": m["name"],
                "family": m["family"],
                "corr_with_rate": m.get("corr_with_rate"),
                "val_f1": val_f1,
                "val_accuracy": m.get("val_accuracy") or m.get("calibrated_accuracy"),
                "val_n": m.get("val_n", 6),
                "loo_f1": m.get("loo_f1"),
                "beats_majority": beats_majority,
                "recommended": m["recommended"],
                "blurb": m.get("blurb", ""),

                "f1": val_f1,
                "accuracy": m.get("val_accuracy") or m.get("calibrated_accuracy"),
            }
        )

    rows.sort(key=lambda r: (-(r["corr_with_rate"] or 0), -(r["val_f1"] or 0)))
    return {
        "models": rows,
        "baseline": {
            "name": "Majority class (always predict deforestation)",
            "val_f1": majority_f1,
            "val_accuracy": majority_acc,
            "note": "Val set has 5/6 positive patches — F1≈90.9% with no learning.",
        },
        "notes": [
            "Metrics are on a held-out validation split (n=6), not the training patches.",
            "Binary F1 often looks ~90–100% because 14/16 patches exceed the 10% deforestation label threshold — the task is heavily imbalanced.",
            "Correlation with continuous deforestation rate is the more meaningful ranking signal (does the score track how much forest was lost?).",
            "Majority-class baseline Val F1 ≈ 90.9%. Models at ~90.9% are not clearly better than always predicting deforestation.",
            "Treat all numbers as demo-scale evidence, not production claims.",
        ],
    }

def _build_model(architecture: str, input_channels: int) -> torch.nn.Module:
    if architecture == "LateFusionCNN":
        return LateFusionCNN()
    if architecture == "RobustBaselineCNN":
        return RobustBaselineCNN(input_channels)
    if architecture == "SimpleCNN":
        return SimpleCNN(input_channels)
    raise ValueError(f"Unknown architecture: {architecture}")

class ModelCache:
    def __init__(self) -> None:
        self.device = torch.device(config.DEVICE)
        self._dl_models: Dict[str, torch.nn.Module] = {}
        self._ml_models: Dict[str, Any] = {}

    def get_meta(self, model_id: str) -> Dict[str, Any]:
        if model_id not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model_id: {model_id}")
        return MODEL_REGISTRY[model_id]

    def get_dl(self, model_id: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        meta = self.get_meta(model_id)
        if meta["family"] != "deep_learning":
            raise ValueError(f"{model_id} is not a deep learning model")
        if model_id not in self._dl_models:
            weights_path = os.path.join(MODELS_DIR, meta["weights"])
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Missing weights: {weights_path}")
            model = _build_model(meta["architecture"], meta["input_channels"])
            state = torch.load(weights_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            self._dl_models[model_id] = model
        return self._dl_models[model_id], meta

    def get_ml(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        meta = self.get_meta(model_id)
        if meta["family"] != "classical_ml":
            raise ValueError(f"{model_id} is not a classical ML model")
        if model_id not in self._ml_models:
            weights_path = os.path.join(MODELS_DIR, meta["weights"])
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Missing weights: {weights_path}")
            with open(weights_path, "rb") as f:
                payload = pickle.load(f)
            self._ml_models[model_id] = payload
        return self._ml_models[model_id], meta

_MODEL_CACHE = ModelCache()

def normalize_array(arr: np.ndarray) -> np.ndarray:
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-8)

def load_raster(path: str) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        bounds = src.bounds
        crs = src.crs.to_string() if src.crs else None
    meta = {
        "profile": profile,
        "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
        "crs": crs,
        "height": data.shape[1],
        "width": data.shape[2],
    }
    return data, meta

def load_rgb_array(path: Optional[str], channels: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if channels is None:
        if not path:
            return None
        data, _ = load_raster(path)
    else:
        data = channels
    if data.shape[0] < 3:
        band = data[0]
        rgb = np.stack([band, band, band], axis=-1)
    else:
        rgb = np.stack([data[0], data[1], data[2]], axis=-1)
    lo, hi = np.percentile(rgb, (2, 98))
    if hi <= lo:
        hi = lo + 1e-6
    rgb = np.clip((rgb - lo) / (hi - lo), 0, 1)
    return (rgb * 255).astype(np.uint8)

def combine_inputs(
    sentinel1: Optional[np.ndarray],
    sentinel2: Optional[np.ndarray],
    use_sentinel1: bool,
    use_sentinel2: bool,
) -> np.ndarray:
    parts = []
    if use_sentinel1:
        if sentinel1 is None:
            raise ValueError("This model requires Sentinel-1 data")
        if sentinel1.shape[0] < config.SENTINEL1_BANDS:
            raise ValueError(f"Sentinel-1 needs {config.SENTINEL1_BANDS} bands")
        parts.append(normalize_array(sentinel1[: config.SENTINEL1_BANDS]))
    if use_sentinel2:
        if sentinel2 is None:
            raise ValueError("This model requires Sentinel-2 data")
        if sentinel2.shape[0] < config.SENTINEL2_BANDS:
            raise ValueError(f"Sentinel-2 needs {config.SENTINEL2_BANDS} bands")
        parts.append(normalize_array(sentinel2[: config.SENTINEL2_BANDS]))
    if not parts:
        raise ValueError("At least one data source is required")
    return np.concatenate(parts, axis=0).astype(np.float32)

def extract_features_from_arrays(
    sentinel1: Optional[np.ndarray],
    sentinel2: Optional[np.ndarray],
    use_sentinel1: bool,
    use_sentinel2: bool,
) -> np.ndarray:
    features: List[float] = []

    def band_stats(band: np.ndarray) -> List[float]:
        flat = band.flatten()
        return [
            float(np.mean(flat)),
            float(np.std(flat)),
            float(np.median(flat)),
            float(np.percentile(flat, 25)),
            float(np.percentile(flat, 75)),
            float(np.min(flat)),
            float(np.max(flat)),
            float(np.var(flat)),
            float(np.ptp(flat)),
        ]

    if use_sentinel1:
        if sentinel1 is None:
            raise ValueError("This model requires Sentinel-1 data")
        for b in range(min(sentinel1.shape[0], config.SENTINEL1_BANDS)):
            features.extend(band_stats(sentinel1[b]))
    if use_sentinel2:
        if sentinel2 is None:
            raise ValueError("This model requires Sentinel-2 data")
        for b in range(min(sentinel2.shape[0], config.SENTINEL2_BANDS)):
            features.extend(band_stats(sentinel2[b]))
    return np.array(features, dtype=np.float64)

def _tile_starts(length: int, tile_size: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, tile_size))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts

def predict_tiled(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> Tuple[np.ndarray, float]:
    _, height, width = image.shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    scores: List[float] = []

    ys = _tile_starts(height, tile_size)
    xs = _tile_starts(width, tile_size)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile = image[:, y : y + tile_size, x : x + tile_size]
                th, tw = tile.shape[1], tile.shape[2]
                if th < 32 or tw < 32:
                    continue
                if th != tile_size or tw != tile_size:
                    padded = np.zeros((image.shape[0], tile_size, tile_size), dtype=np.float32)
                    padded[:, :th, :tw] = tile
                    tile = padded
                tensor = torch.from_numpy(tile).unsqueeze(0).to(device)
                prob = float(model(tensor).item())
                scores.append(prob)
                heatmap[y : y + th, x : x + tw] += prob
                counts[y : y + th, x : x + tw] += 1.0

    counts = np.maximum(counts, 1.0)
    heatmap = heatmap / counts
    whole = float(np.mean(scores)) if scores else 0.0
    return heatmap, whole

def predict_whole_image(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
) -> float:
    with torch.no_grad():
        tensor = torch.from_numpy(image).unsqueeze(0).to(device)
        return float(model(tensor).item())

def save_geotiff(
    path: str,
    array: np.ndarray,
    profile: Optional[dict],
    bounds: Optional[List[float]],
) -> None:
    height, width = array.shape
    if profile:
        out_profile = profile.copy()
        out_profile.update({"count": 1, "dtype": "float32", "height": height, "width": width})
        with rasterio.open(path, "w", **out_profile) as dst:
            dst.write(array.astype(np.float32), 1)
        return

    transform = Affine.identity()
    if bounds and len(bounds) == 4:
        left, bottom, right, top = bounds
        transform = Affine((right - left) / width, 0, left, 0, (bottom - top) / height, top)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(array.astype(np.float32), 1)

def save_heatmap_png(path: str, heatmap: np.ndarray, preview_rgb: Optional[np.ndarray] = None) -> None:
    h, w = heatmap.shape
    if preview_rgb is None:
        base = np.zeros((h, w, 3), dtype=np.uint8)
        base[:, :, 1] = 40
    else:
        if preview_rgb.shape[0] != h or preview_rgb.shape[1] != w:
            preview_img = Image.fromarray(preview_rgb).resize((w, h), Image.BILINEAR)
            base = np.array(preview_img)
        else:
            base = preview_rgb.copy()

    overlay = base.astype(np.float32)
    alpha = np.clip(heatmap, 0, 1)[..., None]
    red = np.array([220, 40, 40], dtype=np.float32)
    overlay = overlay * (1 - 0.65 * alpha) + red * (0.65 * alpha)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)

def save_preview_png(path: str, rgb: np.ndarray) -> None:
    Image.fromarray(rgb).save(path)

def _resolve_inputs(
    sentinel1_path: Optional[str],
    sentinel2_path: Optional[str],
    sentinel1_array: Optional[np.ndarray],
    sentinel2_array: Optional[np.ndarray],
    bounds: Optional[List[float]],
    crs: Optional[str],
    geo_profile: Optional[dict],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[float]], Optional[str], Optional[dict]]:
    s1, s2 = sentinel1_array, sentinel2_array
    local_bounds, local_crs, local_profile = bounds, crs, geo_profile
    if s1 is None and sentinel1_path:
        s1, info = load_raster(sentinel1_path)
        local_bounds = local_bounds or info["bounds"]
        local_crs = local_crs or info["crs"]
        local_profile = local_profile or info["profile"]
    if s2 is None and sentinel2_path:
        s2, info = load_raster(sentinel2_path)
        local_bounds = local_bounds or info["bounds"]
        local_crs = local_crs or info["crs"]
        local_profile = local_profile or info["profile"]
    return s1, s2, local_bounds, local_crs, local_profile

def run_prediction(
    job_id: str,
    model_id: str = "simplecnn_sentinel2",
    sentinel1_path: Optional[str] = None,
    sentinel2_path: Optional[str] = None,
    sentinel1_array: Optional[np.ndarray] = None,
    sentinel2_array: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    tile_size: int = DEFAULT_TILE_SIZE,
    geo_profile: Optional[dict] = None,
    bounds: Optional[List[float]] = None,
    crs: Optional[str] = None,
    patch_idx: Optional[int] = None,
) -> PredictionResult:
    ensure_output_dir()
    meta = _MODEL_CACHE.get_meta(model_id)
    decision_threshold, calibrated = get_model_threshold(model_id, threshold)

    s1, s2, local_bounds, local_crs, local_profile = _resolve_inputs(
        sentinel1_path, sentinel2_path, sentinel1_array, sentinel2_array, bounds, crs, geo_profile
    )

    preview_rgb = load_rgb_array(sentinel2_path, channels=s2 if s2 is not None else s1)
    ref = s2 if s2 is not None else s1
    if ref is None:
        raise ValueError("No imagery provided")
    height = ref.shape[1]
    width = ref.shape[2]

    if meta["family"] == "classical_ml":
        payload, _ = _MODEL_CACHE.get_ml(model_id)
        pipe = payload["pipeline"]
        if patch_idx is not None:
            feats = extract_features_from_patch(
                patch_idx, meta["use_sentinel1"], meta["use_sentinel2"]
            )
        else:
            feats = extract_features_from_arrays(
                s1, s2, meta["use_sentinel1"], meta["use_sentinel2"]
            )
        proba = float(pipe.predict_proba(feats.reshape(1, -1))[0, 1])
        probability = proba
        heatmap = np.full((height, width), proba, dtype=np.float32)
        tile_count = 1
        estimated = probability
    else:
        model, _ = _MODEL_CACHE.get_dl(model_id)
        image = combine_inputs(s1, s2, meta["use_sentinel1"], meta["use_sentinel2"])

        probability = predict_whole_image(model, image, _MODEL_CACHE.device)

        heatmap, _ = predict_tiled(model, image, _MODEL_CACHE.device, tile_size=tile_size)
        tile_count = int(np.ceil(image.shape[1] / tile_size) * np.ceil(image.shape[2] / tile_size))
        estimated = estimate_rate(model_id, probability)

    is_deforested = probability >= decision_threshold
    binary = (heatmap >= decision_threshold).astype(np.float32)
    deforestation_fraction = float(np.mean(binary))
    mean_confidence = float(np.mean(heatmap))

    mask_path = os.path.join(OUTPUT_DIR, f"{job_id}_mask.tif")
    heatmap_path = os.path.join(OUTPUT_DIR, f"{job_id}_heatmap.png")
    preview_path = os.path.join(OUTPUT_DIR, f"{job_id}_preview.png")

    save_geotiff(mask_path, heatmap, local_profile, local_bounds)
    if preview_rgb is not None:
        save_preview_png(preview_path, preview_rgb)
    else:
        preview_path = None
    save_heatmap_png(heatmap_path, heatmap, preview_rgb)

    return PredictionResult(
        job_id=job_id,
        model_id=model_id,
        probability=probability,
        is_deforested=is_deforested,
        threshold=decision_threshold,
        deforestation_fraction=deforestation_fraction,
        mean_confidence=mean_confidence,
        tile_count=tile_count,
        image_size=(height, width),
        mask_path=mask_path,
        heatmap_path=heatmap_path,
        preview_path=preview_path,
        bounds=local_bounds,
        crs=local_crs,
        estimated_rate=estimated,
        model_family=meta["family"],
        calibrated=calibrated,
    )

def predict_demo_patch(
    patch_idx: int,
    model_id: str = "simplecnn_sentinel2",
    threshold: Optional[float] = None,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> PredictionResult:
    if patch_idx < 0 or patch_idx >= config.NUM_PATCHES:
        raise ValueError(f"patch_idx must be in 0..{config.NUM_PATCHES - 1}")
    s1_path = os.path.join(config.SENTINEL1_PATH, f"RASTER_{patch_idx}.tif")
    s2_path = os.path.join(config.SENTINEL2_PATH, f"RASTER_{patch_idx}.tif")
    job_id = f"demo_patch_{patch_idx}_{model_id}"
    return run_prediction(
        job_id=job_id,
        model_id=model_id,
        sentinel1_path=s1_path if os.path.exists(s1_path) else None,
        sentinel2_path=s2_path if os.path.exists(s2_path) else None,
        threshold=threshold,
        tile_size=tile_size,
        patch_idx=patch_idx,
    )

def compare_models_on_patch(
    patch_idx: int,
    model_ids: Optional[List[str]] = None,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> Dict[str, Any]:
    available = {m["id"]: m for m in list_available_models() if m["available"]}
    if model_ids is None:
        model_ids = [
            "simplecnn_sentinel2",
            "latefusion_combined",
            "simplecnn_sentinel1",
            "robustbaseline_combined",
            "randomforest_combined",
            "logisticregression_combined",
            "knn_combined",
        ]
    model_ids = [m for m in model_ids if m in available]

    patches = {p["id"]: p for p in list_demo_patches()}
    if patch_idx not in patches:
        raise ValueError(f"Unknown patch {patch_idx}")
    gt = patches[patch_idx]["ground_truth_deforestation_rate"]
    gt_label = 1 if (gt or 0) > 0.1 else 0

    comparisons = []
    preview_url = None
    for mid in model_ids:
        result = predict_demo_patch(patch_idx, model_id=mid, tile_size=tile_size)
        payload = result_to_dict(result)
        payload["correct"] = int(result.is_deforested) == gt_label
        payload["name"] = available[mid]["name"]
        payload["family"] = available[mid]["family"]
        payload["blurb"] = available[mid].get("blurb", "")
        if preview_url is None and payload.get("preview_url"):
            preview_url = payload["preview_url"]
        comparisons.append(payload)

    comparisons.sort(key=lambda c: -c["probability"])
    return {
        "patch_id": patch_idx,
        "ground_truth_deforestation_rate": gt,
        "ground_truth_label": "Deforestation" if gt_label else "Forest intact",
        "preview_url": preview_url,
        "comparisons": comparisons,
        "agreement": sum(1 for c in comparisons if c["correct"]),
        "total_models": len(comparisons),
    }

def result_to_dict(result: PredictionResult) -> Dict[str, Any]:
    return {
        "job_id": result.job_id,
        "model_id": result.model_id,
        "model_family": result.model_family,
        "probability": round(result.probability, 4),
        "estimated_rate": round(result.estimated_rate, 4),
        "is_deforested": result.is_deforested,
        "threshold": round(result.threshold, 4),
        "calibrated": result.calibrated,
        "deforestation_fraction": round(result.deforestation_fraction, 4),
        "mean_confidence": round(result.mean_confidence, 4),
        "tile_count": result.tile_count,
        "image_height": result.image_size[0],
        "image_width": result.image_size[1],
        "mask_url": f"/outputs/{os.path.basename(result.mask_path)}",
        "heatmap_url": f"/outputs/{os.path.basename(result.heatmap_path)}",
        "preview_url": (
            f"/outputs/{os.path.basename(result.preview_path)}"
            if result.preview_path
            else None
        ),
        "bounds": result.bounds,
        "crs": result.crs,
        "label": "Deforestation likely" if result.is_deforested else "Forest intact",
    }
