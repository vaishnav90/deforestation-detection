# How Our Deforestation Models Work — A Slow Walkthrough

This document teaches **what we built**, **why each step exists**, and **how the pieces connect**, from raw satellite pixels to a live detector. Read it in order. You do not need to be a deep-learning expert; every idea is introduced before it is used.

---

## Table of contents

1. [The goal in plain English](#1-the-goal-in-plain-english)
2. [What the data looks like](#2-what-the-data-looks-like)
3. [Turning pixels into a learning problem](#3-turning-pixels-into-a-learning-problem)
4. [Train vs validation (and why it matters)](#4-train-vs-validation-and-why-it-matters)
5. [Two families of models](#5-two-families-of-models)
6. [Classical ML, step by step](#6-classical-ml-step-by-step)
7. [Deep learning CNNs, step by step](#7-deep-learning-cnns-step-by-step)
8. [How training actually works](#8-how-training-actually-works)
9. [How we measure “good”](#9-how-we-measure-good)
10. [Why 100% F1 was misleading](#10-why-100-f1-was-misleading)
11. [Calibration (making scores usable)](#11-calibration-making-scores-usable)
12. [Inference: using a trained model](#12-inference-using-a-trained-model)
13. [What the website does with the models](#13-what-the-website-does-with-the-models)
14. [Which model to trust for what](#14-which-model-to-trust-for-what)
15. [File map (where the code lives)](#15-file-map-where-the-code-lives)
16. [Glossary](#16-glossary)

---

## 1. The goal in plain English

We want a computer program that looks at a **satellite image of a forest region** and answers:

> “Does this patch look like it has meaningful deforestation?”

More precisely, for each image patch we want:

1. A **score** between 0 and 1 (how strongly the model thinks deforestation is present).
2. A **yes/no decision** (deforested vs mostly intact), by comparing that score to a threshold.
3. Optionally, a **heatmap** showing which parts of a large image look riskier.

We are **not** (in this project) drawing a perfect outline of every cut tree. Most of our models are **patch classifiers**: they look at a whole tile and output one number.

---

## 2. What the data looks like

### 2.1 Two kinds of satellite imagery

| Sensor | Nickname | What it measures | Bands we use | Strength |
|--------|----------|------------------|--------------|----------|
| **Sentinel-2** | Optical | Reflected sunlight (like a camera) | 4: Red, Green, Blue, Near-Infrared | Great detail when skies are clear |
| **Sentinel-1** | Radar | Microwave backscatter | 2: VV, VH polarizations | Works through clouds and at night |

Think of it this way:

- **Sentinel-2** = “What color is the land?”
- **Sentinel-1** = “How rough / structured is the surface?” (even under clouds)

Using both is useful because tropical forests are often cloudy. Optical alone can go blind for weeks; radar still sees something.

### 2.2 Patches, not one giant image

The full scene is huge. For training we use **16 grid patches**, each **2816 × 2816 pixels**:

```
RASTER_0.tif … RASTER_15.tif
```

Each patch has matching files for:

- Sentinel-1 radar  
- Sentinel-2 optical  
- A **mask** (ground-truth labels)

Paths are configured in `config.py` under `Claude AI Archive/`.

### 2.3 The mask (the “answer key”)

The mask is a single-band image the same size as the patch. Roughly:

- Pixel value **1** → that location is labeled **deforested**
- Other values → **not deforested** (forest / other)

From the mask we compute one number per patch:

\[
\text{deforestation rate} = \frac{\text{number of deforested pixels}}{\text{total pixels}}
\]

Example:

- Patch 0 ≈ **3%** deforested → mostly intact  
- Patch 7 ≈ **78%** deforested → heavy loss  

That continuous rate is important later when we talk about **correlation**.

---

## 3. Turning pixels into a learning problem

A neural net cannot “just know” what deforestation means. We must define:

1. **Input** \(X\): the satellite image (or features derived from it)  
2. **Target** \(y\): what we want it to predict  

### 3.1 Our binary label rule

We convert the continuous rate into a **binary class**:

> If deforestation rate **> 10%**, label the patch as **positive** (deforested).  
> Otherwise label it **negative** (intact).

Why 10%? It is a practical threshold used in this project so that “a little noise” does not count as deforestation, but clear loss does.

**Important consequence:**  
Out of 16 patches, **14 are positive** and only **2 are negative**.  
The dataset is **heavily imbalanced**. That will matter when we interpret accuracy and F1.

### 3.2 Normalization

Raw satellite values can be huge or tiny depending on sensor and processing. Before feeding images into a CNN we **normalize** each image:

\[
x_{\text{norm}} = \frac{x - \text{mean}(x)}{\text{std}(x) + \epsilon}
\]

So each input has roughly mean 0 and standard deviation 1. Neural nets train more stably that way.

This happens in `data_loader.py` during training and again in `inference.py` at prediction time. **Training and inference must match.**

---

## 4. Train vs validation (and why it matters)

If you test a model on the same images it trained on, it can **memorize** instead of **learn**. So we split patches:

| Split | Patch indices (this project) | Count |
|-------|------------------------------|-------|
| **Train** | 0, 2, 3, 4, 7, 10, 11, 13, 14, 15 | 10 |
| **Validation** | 1, 5, 6, 8, 9, 12 | 6 |

- **Train:** used to update model weights.  
- **Validation:** held out; used to check generalization and to pick things like decision thresholds.

The split is **stratified** so both sets contain some positive and some negative examples (as much as possible with only 2 negatives total).

With only 16 patches, every number is **demo-scale**, not a large scientific study. We still use proper splits so we do not fool ourselves more than necessary.

---

## 5. Two families of models

We built two families on purpose, so we can **compare** approaches in a presentation:

```
┌─────────────────────────────────────────────┐
│              Same question                  │
│   “Is this patch deforested (>10%)?”        │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
 Classical ML              Deep Learning
 (statistics →             (raw pixels →
  Random Forest,            CNN → score)
  Logistic, KNN)
```

| Family | Input | Idea |
|--------|--------|------|
| **Classical ML** | A short list of numbers (mean, std, percentiles per band) | Compress the image into summary stats, then classify |
| **Deep Learning** | The full multi-band image tensor | Let convolutional layers discover spatial patterns |

Neither family is “always better.” On this tiny dataset, classical ML is surprisingly strong. Deep models shine when you have more data and need spatial structure.

---

## 6. Classical ML, step by step

Code: `simple_ml_baselines.py` (training) and sklearn pipelines saved as `trained_models/*_sklearn.pkl`.

### Step A — Extract features from a patch

A 2816×2816×bands image is millions of numbers. Random Forest cannot eat that raw. So for **each band** we compute statistics such as:

- mean  
- standard deviation  
- median  
- 25th and 75th percentiles  
- min, max  
- variance  
- range (max − min)  

If we use Sentinel-1 (2 bands) + Sentinel-2 (4 bands), we get:

\[
6 \text{ bands} \times 9 \text{ stats} = 54 \text{ features}
\]

That vector is the model’s entire view of the patch.

**Intuition:** deforestation often changes brightness and variability of radar/optical bands. Summaries can capture that even without looking at spatial layout.

### Step B — Choose a classifier

We trained several:

1. **Random Forest** — many decision trees voting; robust and strong on tabular features.  
2. **Logistic Regression** — a linear weighted sum + sigmoid; simple and interpretable.  
3. **K-Nearest Neighbors (KNN)** — “what did similar patches look like in training?”

We wrap each with a **StandardScaler** (zero-mean / unit-variance features) inside a sklearn `Pipeline`, so scaling and the classifier travel together.

### Step C — Train

Fit on the **train** patch feature vectors and their binary labels.

### Step D — Predict

At inference:

1. Compute the same 54 (or fewer) features from the new image.  
2. Run `predict_proba` → probability of class “deforested”.  
3. If probability ≥ 0.5 → predict deforestation.

Classical models do **not** produce a spatial heatmap from pixels; we fill the heatmap with a constant equal to that one probability (the whole patch gets one score).

---

## 7. Deep learning CNNs, step by step

Code: `models.py` (architectures), `train_simple.py` (training), weights in `trained_models/*.pth`.

### 7.1 What a CNN is doing (slowly)

A **convolution** slides a small filter (e.g. 3×3 or 7×7) across the image. Each filter looks for a local pattern (edges, texture, bright spots).

Stacking convolutions:

1. Early layers → simple local patterns  
2. Deeper layers → larger, more abstract patterns  
3. Eventually → a compact summary of the whole patch  

We then map that summary to a single number in \([0,1]\) with a **sigmoid**.

### 7.2 SimpleCNN

Used for:

- **Sentinel-2 only** (4 input channels) — our best overall ranker  
- **Sentinel-1 only** (2 input channels) — cloudy-weather option  

Pipeline inside the network:

1. Conv layers with increasing channels (32 → 64 → 128 → 256)  
2. Strides that shrink the spatial size (the image gets “smaller” but richer)  
3. BatchNorm + Dropout (stabilizes training, reduces overfitting)  
4. **Global average pooling** → collapses height×width to one value per channel  
5. Fully connected layers → 1 output  
6. **Sigmoid** → probability  

Because of global pooling, the network outputs **one score per patch**, not a per-pixel map.

### 7.3 LateFusionCNN (multi-modal)

Idea: radar and optical are different physics. Don’t mash them into one early soup if you can help it.

```
Input (6 bands: S1|S2)
        │
        ├─ first 2 bands ──► Sentinel-1 CNN branch ──► 64 features
        │
        └─ last 4 bands ──► Sentinel-2 CNN branch ──► 64 features
                                    │
                                    ▼
                         concatenate (128 features)
                                    │
                                    ▼
                            fusion MLP + sigmoid
                                    │
                                    ▼
                              one probability
```

This is **late fusion**: each modality gets its own specialist path; we combine high-level features at the end.

### 7.4 RobustBaselineCNN

A deeper CNN with **residual connections** (add the input of a block back to its output). Residuals help gradients flow in deeper nets. In our experiments it was less discriminative on continuous rate (lower correlation) than SimpleCNN on Sentinel-2.

### 7.5 UNetLike (exists, not the main production path)

`models.py` also defines a U-Net-style network that outputs a **spatial map**. We did **not** ship that as the main demo path. The live app’s heatmaps for CNNs come from **tiling a patch classifier**, not from a trained U-Net.

---

## 8. How training actually works

Deep models are trained in `train_simple.py`. Here is the loop in slow motion.

### Step 1 — Load a batch of patches

Each sample is:

- Image tensor shaped `(channels, 2816, 2816)`  
- Mask tensor; we often reduce the mask to a **mean deforestation rate** as the regression-style target for the patch  

### Step 2 — Forward pass

Run the CNN → get a predicted score \(\hat{y} \in (0,1)\).

### Step 3 — Compute loss

Compare prediction to target. The training script uses robust losses (including focal-style ideas for imbalance) so the model does not ignore the rare “intact” class.

### Step 4 — Backward pass

Compute gradients of the loss with respect to every weight (“which knobs should we turn?”).

### Step 5 — Optimizer step

AdamW updates the weights a little bit in the direction that reduces loss (`LEARNING_RATE = 0.0005` in `config.py`).

### Step 6 — Repeat for many epochs

An **epoch** = one full pass over the training set. We allow up to 50 epochs, with **early stopping** if validation stops improving.

### Step 7 — Save the best checkpoint

When validation F1 is best, save:

- `*_best.pth` — model weights  
- `*_checkpoint.pkl` — weights + metadata  
- `*_results.json` — metrics summary  

---

## 9. How we measure “good”

### 9.1 Confusion matrix basics

For binary decisions:

|  | Actually positive | Actually negative |
|--|-------------------|-------------------|
| **Predicted positive** | True Positive (TP) | False Positive (FP) |
| **Predicted negative** | False Negative (FN) | True Negative (TN) |

### 9.2 Metrics we report

- **Accuracy** = (TP+TN) / all  
- **Precision** = TP / (TP+FP) — “when we cry deforestation, how often are we right?”  
- **Recall** = TP / (TP+FN) — “of real deforestation patches, how many did we catch?”  
- **F1** = harmonic mean of precision and recall  

### 9.3 Correlation with deforestation rate

Binary metrics only care about yes/no.  

**Correlation** asks:

> When the true % loss is higher, is the model’s score higher too?

That is often the **more honest ranking signal** on this dataset.  
Example: Simple CNN (Sentinel-2) reaches about **0.97** correlation with the continuous rate on held-out data.

---

## 10. Why 100% F1 was misleading

This is one of the most important lessons in the project.

### The majority-class trap

If 5 of 6 validation patches are “deforested,” a brain-dead rule:

> Always predict deforestation.

…already gets roughly:

- Accuracy ≈ 83%  
- F1 ≈ **90.9%**

So a model with Val F1 = 90.9% may be **no better than always saying “deforested.”**

### What we did about it

1. Report metrics on the **held-out validation split** (n=6), not on train+val mixed together.  
2. Rank models primarily by **correlation with continuous rate**.  
3. Show a **majority baseline** on the leaderboard.  
4. Tag models that only match majority F1 as “≈ majority.”

**Presentation takeaway:**  
Do not brag about 100% F1 on 16 imbalanced patches. Brag about ranking severity (correlation) and about beating the majority baseline.

---

## 11. Calibration (making scores usable)

### The problem we saw

Late Fusion’s raw scores were often stuck around **0.35–0.55**, even when it **ranked** patches well (high correlation).  

If you hard-code “score ≥ 0.5 means deforestation,” Late Fusion looked terrible (very low recall). The model wasn’t useless — the **threshold was wrong**.

### What calibration means here

For each deep model we:

1. Run it on all patches to get raw scores.  
2. On the **training** split only, search for a threshold that maximizes F1.  
3. Save that threshold in `trained_models/model_calibration.json`.  
4. Also fit a simple linear map:

\[
\text{estimated rate} \approx a \cdot \text{raw score} + b
\]

so the UI can show an “estimated deforestation rate,” not only a vague probability.

Examples of calibrated thresholds:

| Model | Optimal threshold (approx.) |
|-------|-----------------------------|
| Simple CNN (S2) | 0.18 |
| Late Fusion | 0.35 |
| Simple CNN (S1) | 0.08 |

At inference, the app uses these thresholds by default (“calibrated threshold”).

---

## 12. Inference: using a trained model

Code: `inference.py`.

### 12.1 Deep model path

1. Load weights once (cached in memory).  
2. Load Sentinel-1 and/or Sentinel-2 GeoTIFFs as needed.  
3. Normalize and stack channels the same way as training.  
4. **Whole-image forward pass** → main probability (matches how we calibrated).  
5. **Tiled forward passes** (e.g. 704×704 tiles) → assemble a **heatmap** for visualization.  
6. Apply calibrated threshold → yes/no label.  
7. Save mask GeoTIFF + PNG overlay.

### 12.2 Why tiling for heatmaps?

The classifier was trained to score a **whole patch**. To visualize *where* risk is higher inside a large image, we:

- cut the image into tiles  
- score each tile  
- paint those scores back onto a map  

That is a **practical visualization**, not the same as a true segmentation model.

### 12.3 Classical ML path

1. Load sklearn pipeline from `.pkl`.  
2. Extract the same statistical features.  
3. `predict_proba` → one score for the whole image.

---

## 13. What the website does with the models

Code: `api.py` + `static/`.

```
Browser (CanopyWatch UI)
        │
        ▼
FastAPI endpoints
  /api/v1/models
  /api/v1/leaderboard
  /api/v1/predict/demo
  /api/v1/compare
  /api/v1/predict   (upload)
        │
        ▼
inference.py  →  trained_models/
        │
        ▼
outputs/  (heatmap PNG, mask GeoTIFF)
```

### Useful modes for demos

1. **Leaderboard** — honest metrics (correlation first).  
2. **Compare models** — run many models on one demo patch (great for slides).  
3. **Single model** — one CNN or ML model + heatmap.  
4. **Upload** — your own GeoTIFFs (must match expected band counts for the chosen model).

Run locally:

```bash
./run.sh
# open http://127.0.0.1:8000
```

---

## 14. Which model to trust for what

| Situation | Prefer | Why |
|-----------|--------|-----|
| Clear-sky optical available | **Simple CNN (Sentinel-2)** | Highest correlation with true loss rate (~0.97) |
| Need something fast & explainable | **Random Forest** | Strong on stats; easy story |
| Cloudy / radar-only | **Simple CNN (Sentinel-1)** | Designed for S1 bands |
| Want multi-modal story | **Late Fusion** | Separate S1/S2 branches; use calibrated threshold |
| Teaching baselines | Logistic / KNN | Show the ladder from simple → complex |

---

## 15. File map (where the code lives)

| File | Role |
|------|------|
| `config.py` | Paths, patch count, batch size, learning rate, band counts |
| `data_loader.py` | Loads GeoTIFFs + masks into PyTorch tensors |
| `models.py` | CNN architectures (SimpleCNN, LateFusion, RobustBaseline, UNetLike) |
| `train_simple.py` | Trains deep models, saves checkpoints |
| `simple_ml_baselines.py` | Feature extraction + classical ML training utilities |
| `inference.py` | Load models, calibrate thresholds, predict, heatmaps, compare |
| `api.py` | FastAPI server |
| `static/` | Website UI + presentation HTML |
| `trained_models/` | `.pth` weights, sklearn `.pkl`, calibration JSON, results |
| `run.sh` | Create venv if needed and start the server |

---

## 16. Glossary

| Term | Meaning |
|------|---------|
| **Band** | One channel of a satellite image (e.g. red, or VV radar) |
| **Patch** | One tile cut from a larger scene (here 2816×2816) |
| **Mask** | Ground-truth label image |
| **Normalization** | Rescaling values so training is stable |
| **CNN** | Convolutional Neural Network — learns from spatial patterns |
| **Epoch** | One full pass through the training set |
| **Overfitting** | Memorizing train data; failing on new data |
| **Threshold** | Cutoff that turns a score into yes/no |
| **Calibration** | Choosing a better threshold / mapping for usable decisions |
| **F1 score** | Balance of precision and recall |
| **Correlation** | How well scores track continuous deforestation % |
| **Late fusion** | Process modalities separately, combine at the end |
| **Heatmap** | Spatial picture of model scores (here from tiling) |
| **Imbalanced data** | One class much more common than the other |

---

## End-to-end story (one paragraph)

We took 16 Sentinel-1/Sentinel-2 patches with deforestation masks, labeled a patch “deforested” if more than 10% of its pixels were cut, trained both classical models on band statistics and CNNs on raw imagery, discovered that raw 0.5 thresholds and all-data F1 scores were misleading on this tiny imbalanced set, calibrated thresholds on the training split, ranked models by correlation with continuous loss rate, and wrapped everything in a FastAPI + web UI so you can compare models live on demo patches.

---

## Suggested study path

If you want to **learn by reading code**, go in this order:

1. `config.py` — what exists on disk  
2. `data_loader.py` — how one training sample is built  
3. `models.py` — `SimpleCNN.forward`, then `LateFusionCNN.forward`  
4. `train_simple.py` — the training loop and metrics  
5. `simple_ml_baselines.py` — `extract_features_from_patch`  
6. `inference.py` — `run_prediction` and `compare_models_on_patch`  
7. `api.py` — how the browser triggers those functions  

That is the full modeling journey of this project, from pixels to a working detector.
