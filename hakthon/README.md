# Smart Farming - Tomato Disease Detection System

An AI-powered web platform for **tomato disease detection** and **smart crop recommendation**, built for the Smart Farming Hackathon. The system combines state-of-the-art computer vision models, a bilingual Arabic/English chatbot, voice transcription, and an LSTM-based crop recommendation engine, all served through a high-performance FastAPI backend.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Disease Classes](#disease-classes)
5. [Models](#models)
6. [Project Structure](#project-structure)
7. [Installation](#installation)
8. [Configuration](#configuration)
9. [Training](#training)
10. [Evaluation](#evaluation)
11. [Export to ONNX](#export-to-onnx)
12. [Running the Web Server](#running-the-web-server)
13. [API Endpoints](#api-endpoints)
14. [Utility Scripts](#utility-scripts)
15. [Technologies and Dependencies](#technologies-and-dependencies)
16. [AI Services and API Keys](#ai-services-and-api-keys)
17. [Data Augmentation Strategy](#data-augmentation-strategy)
18. [Training Strategy](#training-strategy)

---

## Project Overview

This project is a full-stack AI system that helps farmers detect tomato plant diseases from a photo taken on any device. The farmer uploads an image through the web interface; the system:

1. **Verifies** the image contains a tomato plant (via Gemini Vision API).
2. **Classifies** the disease using a fine-tuned Vision Transformer or EfficientNetV2.
3. **Returns** detailed Arabic treatment recommendations (chemical, physical, biological, agricultural).
4. Offers a **plant chatbot** powered by Groq (Llama 3.3 70B) for follow-up questions.
5. Supports **voice input** via Groq Whisper audio transcription.
6. Provides **crop recommendations** based on GPS coordinates and soil data using an LSTM model.

---

## Key Features

| Feature | Description |
|---|---|
| Disease Detection | Detects 10 tomato conditions from a single leaf/fruit image |
| Bilingual | Full Arabic and English support across UI and chatbot |
| Plant Chatbot | Groq-powered Llama 3.3 70B chatbot restricted to plant/farming topics |
| Voice Input | Groq Whisper `whisper-large-v3` audio-to-text transcription |
| Crop Recommendation | GPS-based LSTM crop recommender (Marsa Matrouh region) |
| Tomato Validator | Gemini 1.5 Flash vision check before inference |
| Rich Metrics | Top-1/3/5 accuracy, ECE calibration, per-class F1, confusion matrix |
| ONNX Export | Production ONNX model for cross-platform deployment |
| Gradio UI | Quick-test UI for production model (port 7860) |

---

## System Architecture

```
+----------------------------------------------------------+
|                    FastAPI Web Server                    |
|                   (app.py - port 8000)                   |
+---------------+---------------+--------------------------+
|  /predict     |   /chat       |  /transcribe             |
|  Image Upload |  Plant Bot    |  Whisper STT             |
|               |  (Groq LLM)   |  (Groq Whisper)          |
+---------------+---------------+--------------------------+
|              /api/predict_crop                           |
|              LSTM Crop Recommender (GPS / Auger ID)      |
+------------------------------+---------------------------+
                               |
          +--------------------+--------------------+
          |                    |                    |
  +-------+------+   +---------+-------+  +---------+------+
  | ViT           |   | ViT / EffNetV2  |  | disease_arabic |
  |  (tomato      |   |  Disease Model  |  |    .json       |
  |   check)      |   |  (timm/PyTorch) |  | (AR treatment  |
  +--------------+   +-----------------+  |   database)    |
                                          +----------------+
```

---

## Disease Classes

The model recognises **10 classes** (9 diseases + 1 healthy):

| # | Key | Arabic Name |
|---|-----|-------------|
| 1 | `tomato_bacterial_spot` | التبقع البكتيري |
| 2 | `tomato_early_blight` | اللفحة المبكرة |
| 3 | `tomato_healthy` | نبات سليم |
| 4 | `tomato_late_blight` | اللفحة المتأخرة |
| 5 | `tomato_leaf_mold` | عفن الأوراق |
| 6 | `tomato_mosaic_virus` | فيروس موزاييك الطماطم |
| 7 | `tomato_septoria_leaf_spot` | تبقع الأوراق السبتوري |
| 8 | `tomato_spider_mites` | سوس العنكبوت |
| 9 | `tomato_target_spot` | البقعة الهدفية |
| 10 | `tomato_yellow_leaf_curl_virus` | فيروس تجعد واصفرار الأوراق (TYLCV) |

For each disease, `disease_arabic.json` stores:
- **Symptoms** (3-5 bullet points in Arabic)
- **Chemical control** recommendations
- **Physical control** methods
- **Biological control** options
- **Agricultural practices**

---

## Models

### 1. ViT - Vision Transformer (`train.py`)

| Setting | Value |
|---|---|
| Backbone | `vit_large_patch16_224` (via timm) |
| Input size | 224 x 224 |
| Classes | 10 |
| Dropout | 0.2 |
| Training stages | 3 progressive stages |
| Pretrained | ImageNet |
| Mixed precision | AMP enabled |

### 2. EfficientNetV2-S - Production Model (`train_production.py`)

| Setting | Value |
|---|---|
| Backbone | `tf_efficientnetv2_s.in21k_ft_in1k` |
| Input size | 224 x 224 |
| Classes | 10 |
| Dropout | 0.3 |
| Loss | Focal Loss (gamma=2.0, alpha=class-weighted) |
| Sampler | WeightedRandomSampler |
| Training stages | 3 progressive stages |
| Min confidence | 0.70 |
| ONNX export | Yes (`export_onnx.py`) |

---

## Project Structure

```
hakthon/
|
|-- app.py                   # FastAPI application - main entry point
|-- config.py                # All hyperparameters, paths and settings (ViT)
|-- train.py                 # ViT progressive fine-tuning trainer
|-- train_production.py      # EfficientNetV2 production trainer (self-contained)
|-- evaluate.py              # Standalone evaluation script with rich metrics
|-- export_onnx.py           # Export best_production.pth to ONNX
|-- benchmark.py             # Single-epoch timing benchmark
|-- test_ui.py               # Gradio quick-test UI (port 7860)
|-- test_vision.py           # Async Groq vision API smoke-test
|-- patch_keras.py           # Fix DTypePolicy in legacy .keras files
|-- translate_json.py        # Translate disease JSON to Arabic via Groq LLM
|-- disease_arabic.json      # Arabic disease treatment database (10 classes)
|-- requirements.txt         # Python dependencies
|
|-- CropREC/                 # Crop recommendation sub-module (LSTM model)
|
|-- src/                     # Core source modules (imported by scripts above)
    |-- inference.py             # ViT predictor (create_predictor)
    |-- inference_production.py  # EfficientNetV2 predictor
    |-- model_architecture.py    # ViT model builder (create_model)
    |-- data_pipeline.py         # DataLoaders + Albumentations transforms
    |-- trainer.py               # ProgressiveTrainer (3-stage, ViT)
    |-- dataset_analyzer.py      # Dataset statistics and class counts
    |-- crop_recommender.py      # LSTM crop recommendation logic
    |-- utils.py                 # Metrics, plotting, checkpoint helpers
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (NVIDIA recommended, e.g. RTX 4060)
- CUDA 11.8+ / cuDNN 8+

### Steps

```bash
# 1. Navigate to the project directory
cd hakthon

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install Gradio for the test UI
pip install gradio

# 5. Set up environment variables
echo GROQ_API_KEY=your_groq_key_here > .env
```

---

## Configuration

All ViT training settings live in **`config.py`**. Key sections:

### Paths

```python
BASE_DIR    = Path("/home/smart-farming")
DATA_DIR    = BASE_DIR / "tomato_extract/.../PlantDiseased"
MODELS_DIR  = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
```

> **Windows users:** Update `BASE_DIR` in `config.py` and `train_production.py` to your local dataset path.

### Training Stages (ViT)

| Stage | Epochs | LR | Batch Size | What trains |
|---|---|---|---|---|
| 1 | 15 | 2e-3 | 32 | Classifier head only |
| 2 | 20 | 5e-5 | 32 | Top 6 transformer blocks + head |
| 3 | 25 | 5e-6 | 24 | Full model |

### Training Stages (EfficientNetV2-S)

| Stage | Epochs | LR | What trains |
|---|---|---|---|
| 1 | 5 | 1e-3 | Head only |
| 2 | 15 | 5e-5 | Top 4 blocks + head |
| 3 | 20 | 1e-5 | Full model |

### Other Key Settings

```python
IMAGE_SIZE               = 224     # Input resolution
DEVICE                   = "cuda"  # auto-falls back to CPU
USE_AMP                  = True    # Mixed precision (FP16)
RANDOM_SEED              = 42
EARLY_STOPPING_PATIENCE  = 7
GRAD_CLIP_MAX_NORM       = 0.5
MIN_CONFIDENCE_THRESHOLD = 0.3     # Below this -> "uncertain"
```

---

## Training

### Option A - ViT (`train.py`)

```bash
# Full 3-stage training
python train.py

# Custom epochs per stage
python train.py --stage1-epochs 10 --stage2-epochs 15 --stage3-epochs 20

# Skip a stage
python train.py --skip-stage1

# Debug mode (2 epochs per stage)
python train.py --debug

# CPU training
python train.py --device cpu --no-amp
```

**All CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--stage1-epochs` | 15 | Epochs for stage 1 |
| `--stage2-epochs` | 20 | Epochs for stage 2 |
| `--stage3-epochs` | 25 | Epochs for stage 3 |
| `--skip-stage1/2/3` | False | Skip a stage entirely |
| `--model-name` | `vit_large_patch16_224` | timm model name |
| `--batch-size-stage1/2/3` | 32/32/24 | Batch sizes |
| `--num-workers` | 4 | DataLoader workers |
| `--device` | cuda | `cuda` or `cpu` |
| `--no-amp` | False | Disable mixed precision |
| `--seed` | 42 | Random seed |
| `--debug` | False | 2-epoch quick test |
| `--use-balanced-sampling` | False | WeightedRandomSampler |

### Option B - EfficientNetV2 Production (`train_production.py`)

```bash
python train_production.py
```

This script is fully self-contained (no `config.py` dependency). It runs all 3 stages automatically with:
- **Focal Loss** (gamma=2.0) with inverse-frequency class weights
- **WeightedRandomSampler** for balanced mini-batches
- **Cosine Annealing** with linear warmup
- **Early stopping** (patience=10)
- Saves `best_production.pth` and `stage{N}_final.pth` checkpoints

**Output files:**

```
/home/smart-farming/models/production/
  best_production.pth          <- Best checkpoint
  stage1_final.pth
  stage2_final.pth
  stage3_final.pth
  classification_report.txt
  model_metadata.json

/home/smart-farming/outputs/plots_production/
  confusion_matrix_production.png
  training_curves_production.png
```

---

## Evaluation

```bash
python evaluate.py --model-path /home/smart-farming/models/best_model.pth

# With prediction visualizations saved
python evaluate.py --model-path /path/to/model.pth --save-predictions --batch-size 64
```

**Metrics reported:**
- Overall accuracy
- Per-class Precision / Recall / F1-Score
- Macro and Weighted averages
- **Top-3 accuracy**
- **Top-5 accuracy**
- Average confidence for correct vs incorrect predictions
- **Expected Calibration Error (ECE)** - 10 bins

**Output files:**

```
outputs/
  evaluation_report.json
  plots/
    evaluation_confusion_matrix.png
    evaluation_predictions.png     (if --save-predictions)
```

---

## Export to ONNX

```bash
python export_onnx.py
```

Requires `best_production.pth` to exist. Exports:

```
models/production/
  tomato_disease_v2.onnx    <- Cross-platform inference model
  onnx_metadata.json        <- Class names, normalization stats, min_confidence
```

- ONNX input: `image` — shape `[batch, 3, 224, 224]`
- ONNX output: `logits` — shape `[batch, 10]`

---

## Running the Web Server

```bash
python app.py
```

Server starts at `http://0.0.0.0:8000`

| URL | Description |
|---|---|
| `http://localhost:8000/` | Main web UI |
| `http://localhost:8000/docs` | Swagger / OpenAPI documentation |
| `http://localhost:8000/health` | Health check |

### Gradio Test UI (optional)

```bash
python test_ui.py
```

Starts at `http://0.0.0.0:7860` - drag-and-drop interface for the production model.

---

## API Endpoints

### `GET /health`

```json
{ "status": "healthy", "model_loaded": true }
```

---

### `POST /predict`

Upload a tomato image for disease detection.

**Form params:**

| Param | Type | Default | Description |
|---|---|---|---|
| `file` | image file | required | JPG/PNG/BMP/WebP, max 10 MB |
| `lang` | string | `"ar"` | `"ar"` for Arabic, `"en"` for English |

**Success response (diseased):**

```json
{
  "success": true,
  "is_tomato": true,
  "disease_name": "اللفحة المبكرة (Early Blight)",
  "confidence": 0.93,
  "treatment": {
    "symptoms": ["..."],
    "prevention_and_control": {
      "chemical_control": "...",
      "physical_control": "...",
      "biological_control": "...",
      "agricultural_practices": "..."
    }
  },
  "uploaded_image": "/path/to/image.jpg"
}
```

**Non-tomato image response:**

```json
{
  "success": true,
  "is_tomato": false,
  "message": "عذراً، لم يتم التعرف على نبات طماطم في هذه الصورة."
}
```

---

### `POST /chat`

Plant-specialized Arabic/English chatbot.

**Request body:**

```json
{
  "message": "ما هي أعراض اللفحة المبكرة؟",
  "history": [
    { "role": "user", "content": "مرحبا" },
    { "role": "assistant", "content": "مرحباً! كيف أستطيع مساعدتك؟" }
  ]
}
```

**Response:**

```json
{
  "reply": "اللفحة المبكرة تظهر كبقع بنية صغيرة...",
  "success": true
}
```

> The chatbot uses Groq's `llama-3.3-70b-versatile` and enforces a strict plant/agriculture topic boundary. Non-plant questions are politely refused. Conversation history (last 10 turns) is included for context.

---

### `POST /transcribe`

Transcribe audio to text using Groq Whisper.

**Form params:**

| Param | Type | Default | Description |
|---|---|---|---|
| `file` | audio file | required | Any audio format |
| `lang` | string | `"ar"` | Transcription language hint |

**Response:**

```json
{ "text": "ما هو علاج اللفحة المتأخرة؟", "success": true }
```

---

### `POST /api/predict_crop`

Crop recommendation based on location or soil auger ID.

**Request body (GPS):**

```json
{
  "lat": 31.2,
  "lon": 27.1,
  "date": "2026-05-07",
  "max_distance_km": 15.0
}
```

**Request body (Auger ID):**

```json
{
  "auger_id": 14,
  "date": "2026-05-07",
  "depth": "0-30cm"
}
```

**Response:**

```json
{
  "status": "ok",
  "resolved_from": "gps",
  "auger_id": 14,
  "distance_km": 1.5,
  "date_used": "2026-05-07",
  "lstm": {
    "recommended_crops": ["الزيتون", "التين", "الشعير"],
    "scores": {
      "الزيتون": 0.61,
      "التين": 0.25,
      "الشعير": 0.14
    }
  }
}
```

> If the GPS location is outside the mapped region (distance > `max_distance_km`), returns `"status": "out_of_region"`.

---

## Utility Scripts

### `benchmark.py`
Measures training throughput (time per epoch) on the ViT model.
```bash
python benchmark.py
```

### `test_vision.py`
Async smoke-test for Groq vision API (checks if a given image contains a tomato).
```bash
python test_vision.py
```

### `translate_json.py`
Translates a disease knowledge JSON file from English to Arabic using Groq LLM.
```bash
# Place your English JSON as "disease_update (1).json" in the same directory
python translate_json.py
# Output: disease_arabic.json
```

### `patch_keras.py`
Patches `DTypePolicy` serialization issues in `.keras` model files created by Keras 3.3+. Used for the LSTM crop model.
```bash
python patch_keras.py
```

---

## Technologies and Dependencies

### Core ML Stack

| Library | Version | Purpose |
|---|---|---|
| PyTorch | >= 2.0.0 | Deep learning framework |
| torchvision | >= 0.15.0 | Image utilities |
| timm | >= 0.9.0 | ViT and EfficientNetV2 pretrained models |
| albumentations | >= 1.3.0 | Advanced image augmentation |
| opencv-python | >= 4.8.0 | Image loading (BGR to RGB) |
| scikit-learn | >= 1.3.0 | Metrics, classification report |

### Web Framework

| Library | Version | Purpose |
|---|---|---|
| FastAPI | >= 0.100.0 | REST API server |
| uvicorn | >= 0.23.0 | ASGI server |
| python-multipart | >= 0.0.6 | File uploads |
| jinja2 | >= 3.1.0 | HTML templating |

### AI / LLM Services

| Library | Purpose |
|---|---|
| `groq` | Groq Python SDK (chat + Whisper) |
| `google-generativeai` | Gemini Vision API |

### Visualization and Logging

| Library | Purpose |
|---|---|
| matplotlib | Training curves, confusion matrix |
| seaborn | Enhanced plots |
| tqdm | Progress bars |
| tensorboard | Training metrics dashboard (optional) |
| colorlog | Colored console logging |

### Utilities

| Library | Purpose |
|---|---|
| pyyaml | YAML config files |
| python-dotenv | `.env` file loading |
| numpy | Numerical operations |
| pandas | Data handling |
| scipy | Scientific utilities |

---

## AI Services and API Keys

| Service | Used For | How to configure |
|---|---|---|
| **Groq** | Chatbot (Llama 3.3 70B) + Whisper STT + JSON translation | Set `GROQ_API_KEY` in `.env` or `app.py` |
| **Google Gemini** | Tomato vs. non-tomato image validation (Gemini 1.5 Flash) | Set `GEMINI_API_KEY` in `app.py` (line 24) |

> **Security Note:** The current code has API keys hardcoded in `app.py`. Before sharing or deploying, move all keys to a `.env` file and load them with `python-dotenv`.

Example `.env` file:
```env
GROQ_API_KEY=gsk_your_key_here
GEMINI_API_KEY=AIza_your_key_here
```

---

## Data Augmentation Strategy

The training pipeline uses **Albumentations** to simulate real-world farm photography conditions:

| Augmentation | Probability | Purpose |
|---|---|---|
| Random Brightness/Contrast | 80% | Varying lighting conditions |
| Hue/Saturation/Value shift | 60% | Color variance in field photos |
| Random Shadow | 40% | Partial shade from sunlight |
| Motion / Gaussian Blur | 35% | Camera shake, distance |
| Gaussian Noise | 35% | Sensor noise simulation |
| Perspective Transform | 35% | Different shooting angles |
| Coarse Dropout | 50% | Occlusion, dirt on lens |
| JPEG Compression | 35% | Low-quality phone cameras |
| Horizontal Flip | 50% | Orientation invariance |
| Vertical Flip | 30% | Orientation invariance |
| Rotation (+/- 30 deg) | 30% | Leaf angle variation |
| Shift/Scale/Rotate | 50% | Position and scale variation |

---

## Training Strategy

The system uses **3-stage progressive fine-tuning** to prevent catastrophic forgetting:

```
Stage 1: Freeze backbone -> Train classifier head only
         (Fast convergence on high LR)
         |
         v
Stage 2: Unfreeze top N transformer/conv blocks
         (Adapt domain-specific features, medium LR)
         |
         v
Stage 3: Unfreeze entire model
         (Full fine-tuning at very low LR)
```

Additional techniques:
- **Focal Loss** (EfficientNetV2 only): Emphasizes hard, misclassified samples
- **Inverse-frequency class weights**: Handles imbalanced class distributions
- **WeightedRandomSampler**: Ensures balanced class representation per batch
- **Cosine Annealing + Linear Warmup**: Smooth LR schedule
- **Gradient clipping** (max norm 0.5): Training stability
- **Early stopping** (patience 7-10): Prevents overfitting

---

## CropREC Sub-module

The `CropREC/` directory contains the crop recommendation engine. It uses:
- **LSTM model** trained on time-series soil data from the Marsa Matrouh region (Egypt)
- **Auger-point mapping**: GPS coordinates are matched to the nearest soil sampling point
- **Seasonal anchoring**: Recommendations are date-aware

---


