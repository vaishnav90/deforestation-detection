from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from inference import (
    DEFAULT_TILE_SIZE,
    OUTPUT_DIR,
    compare_models_on_patch,
    ensure_output_dir,
    get_leaderboard,
    list_available_models,
    list_demo_patches,
    predict_demo_patch,
    result_to_dict,
    run_prediction,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"

app = FastAPI(
    title="Deforestation AI Detector",
    description="Multi-model deforestation detection on Sentinel-1 / Sentinel-2 imagery",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup() -> None:
    ensure_output_dir()
    UPLOAD_DIR.mkdir(exist_ok=True)
    STATIC_DIR.mkdir(exist_ok=True)

@app.get("/api/v1/health")
def health():
    return {"status": "ok", "service": "deforestation-ai-detector", "version": "1.1.0"}

@app.get("/api/v1/models")
def get_models():
    return {"models": list_available_models()}

@app.get("/api/v1/leaderboard")
def leaderboard():
    return get_leaderboard()

@app.get("/api/v1/demo/patches")
def get_demo_patches():
    return {"patches": list_demo_patches()}

@app.post("/api/v1/predict/demo")
def predict_demo(
    patch_id: int = Form(...),
    model_id: str = Form("simplecnn_sentinel2"),
    threshold: Optional[float] = Form(None),
    tile_size: int = Form(DEFAULT_TILE_SIZE),
):
    try:
        result = predict_demo_patch(
            patch_idx=patch_id,
            model_id=model_id,
            threshold=threshold,
            tile_size=tile_size,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    payload = result_to_dict(result)
    patches = {p["id"]: p for p in list_demo_patches()}
    if patch_id in patches:
        payload["ground_truth_deforestation_rate"] = patches[patch_id][
            "ground_truth_deforestation_rate"
        ]
    return payload

@app.post("/api/v1/compare")
def compare(
    patch_id: int = Form(...),
    model_ids: Optional[str] = Form(None),
    tile_size: int = Form(DEFAULT_TILE_SIZE),
):
    ids: Optional[List[str]] = None
    if model_ids:
        ids = [m.strip() for m in model_ids.split(",") if m.strip()]
    try:
        return compare_models_on_patch(patch_id, model_ids=ids, tile_size=tile_size)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Compare failed: {exc}") from exc

@app.post("/api/v1/predict")
async def predict(
    model_id: str = Form("simplecnn_sentinel2"),
    threshold: Optional[float] = Form(None),
    tile_size: int = Form(DEFAULT_TILE_SIZE),
    sentinel1: Optional[UploadFile] = File(None),
    sentinel2: Optional[UploadFile] = File(None),
):
    if sentinel1 is None and sentinel2 is None:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one GeoTIFF (sentinel1 and/or sentinel2)",
        )

    job_id = uuid.uuid4().hex[:12]
    s1_path = None
    s2_path = None

    try:
        if sentinel1 is not None:
            s1_path = str(UPLOAD_DIR / f"{job_id}_s1.tif")
            with open(s1_path, "wb") as f:
                shutil.copyfileobj(sentinel1.file, f)
        if sentinel2 is not None:
            s2_path = str(UPLOAD_DIR / f"{job_id}_s2.tif")
            with open(s2_path, "wb") as f:
                shutil.copyfileobj(sentinel2.file, f)

        result = run_prediction(
            job_id=job_id,
            model_id=model_id,
            sentinel1_path=s1_path,
            sentinel2_path=s2_path,
            threshold=threshold,
            tile_size=tile_size,
        )
        return result_to_dict(result)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

@app.get("/outputs/{filename}")
def get_output(filename: str):
    path = Path(OUTPUT_DIR) / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if path.resolve().parent != Path(OUTPUT_DIR).resolve():
        raise HTTPException(status_code=400, detail="Invalid path")
    media = "image/png" if filename.endswith(".png") else "application/octet-stream"
    return FileResponse(path, media_type=media)

if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

def main() -> None:
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
