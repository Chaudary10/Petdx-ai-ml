# PetDx AI — Stage-1 Animal Image Validator

Part of **PetDx AI**, a Final Year Project (University of Sargodha) building an Android app that detects skin diseases in cats, dogs, cows, and goats from a photo.

This repo contains the **Stage-1 Validator** — a binary gatekeeper model that runs *before* any disease prediction. Its only job is to answer:

> "Is this a valid image of an animal?"

It does **not** identify species and does **not** diagnose disease — that's handled by separate Stage-2 per-species models (not in this repo).

```
Uploaded image
      │
      ▼
 Stage-1 Validator (this repo)
      │
      ├── NEGATIVE → "Please upload a valid animal image."
      │
      └── VALID ANIMAL → passed to Stage-2 species-specific disease model
```

## What's in this repo

| File | Description |
|---|---|
| `notebooks/validator.ipynb` | Full Colab notebook: dataset prep, training, evaluation, TFLite conversion, and stress testing |
| `models/petdx_validator_v2_best.keras` | Trained Keras model (11.6 MB) |
| `models/petdx_validator_v2.tflite` | TFLite-converted model for Android (2.6 MB) — **use this one for the app** |

## Model details

- **Architecture:** MobileNetV2 (ImageNet pretrained) + GlobalAveragePooling2D → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → Dense(1, sigmoid)
- **Input:** 224×224×3, RGB
- **Output:** single sigmoid probability
  - `>= 0.5` → **VALID ANIMAL**
  - `< 0.5` → **NEGATIVE**
- **Class order:** `0 = negative`, `1 = valid_animal`

## ⚠️ Critical: preprocessing rules (read before integrating)

The model has MobileNetV2's `preprocess_input` **built into it** as an internal `Lambda` layer.

**Do NOT apply any extra normalization or preprocessing before feeding the image in.** Doing so double-preprocesses the image and produces wrong predictions (this bug was hit and fixed during development — see notebook Section 11).

Correct pipeline:

```
Image (any source: camera / gallery)
      │
      ▼
Convert to RGB
      │
      ▼
Resize to 224×224          ← direct resize, NOT letterbox (see "Known limitations")
      │
      ▼
Feed as float32 array, shape (1, 224, 224, 3), raw pixel values 0–255
      │
      ▼
model / interpreter (preprocessing happens internally)
      │
      ▼
Sigmoid probability → threshold at 0.5
```

For Android (TFLite Interpreter), input tensor shape is `[1, 224, 224, 3]`, dtype `float32`. No manual normalization step needed — just resize and cast to float.

## Performance

Trained on an improved dataset of 2,132 images (1,242 valid animal + 890 negative, including 540 hard-negative images scraped from Wikimedia Commons across 9 categories: people, vehicles, documents, electronics, objects, buildings, clothing, food, scenery).

**Test set (214 images, unseen):**

| Metric | Score |
|---|---|
| Accuracy | 98.60% |
| Precision | 99.19% |
| Recall | 98.40% |

|  | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.978 | 0.989 | 0.983 | 89 |
| Valid Animal | 0.992 | 0.984 | 0.988 | 125 |

## Stress testing

Beyond the benchmark test set, the model was stress-tested against real-world images that fooled an earlier version (V1) — cars, certificates, people, phones, buildings, scenery, screenshots — plus edge cases specifically targeting extreme aspect ratios (very tall and very wide images) to check whether direct 224×224 resizing (without letterbox padding) causes distortion-related misclassifications.

**Result: aspect ratio is not a real failure mode.** The model correctly classified images across a wide range of ratios, including a very tall screenshot (AR 0.56), a very wide panorama (AR 3.0), and tall animal photos — direct resize is sufficient, no letterboxing needed.

## Known limitations

- **Memes / text-overlay images:** The model can misclassify memes (images with a real or composite animal plus caption text) as valid animal, since this content type wasn't represented in the hard-negative training categories. Considered low-priority — unlikely for real users of a pet skin-diagnosis app to upload memes.
- **Out-of-scope species:** The validator was trained to accept the 4 supported species (cat, dog, cow, goat) but will also pass other real animals (e.g., a cougar) as "valid animal," since it only checks "is this an animal," not species identity. Stage-2 species models should handle unsupported species gracefully.

## Team

- Abdullah Khan — Team Lead
- Ghazala Sultana
- Alia Rubab
- Alishba Noor

Supervisor: Ms. Rameeza Shaheen
