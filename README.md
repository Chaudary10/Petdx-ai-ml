# PetDx AI: ML Models and Android Integration Guide

Part of **PetDx AI**, a Final Year Project (University of Sargodha): an Android app that screens for **skin diseases** in cats, dogs, cows and goats from a photo.

This repo holds every trained model, the training notebooks, and the rules the app must follow so the models behave the way they were tested.

> **Read sections 3 and 4 before writing any app code.** A preprocessing mistake does not crash the app. It silently returns wrong answers and looks like a broken model.

---

## 1. How the app uses the models

```
User picks an animal (Cat / Dog / Goat / Cow)
      │
      ▼
User takes or uploads a photo
      │
      ▼
Stage 1: Validator  (models/validator)
      │
      ├── probability < 0.5  →  "Please upload a valid animal image."  (stop)
      │
      └── probability >= 0.5 →  continue
                                   │
                                   ▼
                    Stage 2: disease model for the animal the user picked
                                   │
                                   ├── top probability < threshold → "unsure" screen
                                   └── otherwise → show class + confidence + disclaimer
```

Key points:

- **Route by the animal the user picked.** Never run a photo through another animal's model. The validator only checks "is this an animal"; it does **not** know the species. If a user picks Cat and uploads a dog, nothing in the pipeline catches it.
- Each Stage-2 model is a **closed set**. It has no "other" class, so anything outside its classes (eye, hoof, udder problems, other diseases, wrong species) is still forced into one of its labels, sometimes with high confidence. The photo guidance (section 6) and the disclaimers are the only protection.
- The validator also lets other real animals through (for example a cougar). Stage 2 cannot detect that either.

---

## 2. What is in this repo

```
models/
  validator/  petdx_validator_v2.tflite   petdx_validator_v2_best.keras
  cat/        cat_skin_disease_model.tflite   cat_skin_disease_labels.txt
  dog/        dog_v2_float16.tflite           dog_labels.txt
  goat/       goat_skin_disease_model.tflite  goat_skin_disease_labels.txt
  cow/        cow_model_v1.tflite  cow_model_v1.keras  cow_config.json
              cow_model_v1_deployment_guide.md
notebooks/
  validator.ipynb   Goat.ipynb
```

### Which file goes in the app

| Model | Ship in the app | Labels |
|---|---|---|
| Validator | `models/validator/petdx_validator_v2.tflite` | none (single output) |
| Cat | `models/cat/cat_skin_disease_model.tflite` | `cat_skin_disease_labels.txt` |
| Dog | `models/dog/dog_v2_float16.tflite` | `dog_labels.txt` |
| Goat | `models/goat/goat_skin_disease_model.tflite` | `goat_skin_disease_labels.txt` |
| Cow | `models/cow/cow_model_v1.tflite` | `cow_config.json` (see the class order in section 3) |

The `.keras` files are for re-exporting only. **Do not ship them in the app.**

For the Android Studio (Java) walkthrough, see **`ANDROID_INTEGRATION_GUIDE.md`**.

---

## 3. Model contract (the numbers the app must match)

All five models share the same input format:

| Item | Value |
|---|---|
| Input tensor | `[1, 224, 224, 3]`, `float32`, **RGB** |
| Pixel values | **Raw 0 to 255. Do NOT divide by 255. No mean/std normalisation.** Every model scales the pixels internally |
| Resize | Whole photo stretched to 224 x 224, bilinear. **No center-crop, no padding (no letterbox)** |

The scaling is built into each model. We checked the graphs of the shipped `.tflite` files: the validator, goat, cat, cow and dog graphs all start with their own rescale/normalise operations right after the input. Adding another normalisation in the app applies it twice and breaks the predictions. (This exact bug was found and fixed during validator development.)

### Outputs and class order

| Model | Output shape | Class order (index 0, 1, 2 ...) | Architecture | Test accuracy |
|---|---|---|---|---|
| Validator | `[1, 1]` sigmoid | `>= 0.5` valid animal, `< 0.5` negative | MobileNetV2 | 98.60% |
| Cat | `[1, 5]` softmax | `Flea_Allergy`, `Health`, `Ringworm`, `Scabies`, `dermatitis` | EfficientNetB0 | 78.46% |
| Dog | `[1, 5]` softmax | `Dermatitis`, `Healthy`, `Hypersensitivity`, `demodicosis`, `ringworm` | MobileNetV2 | 89.1% |
| Goat | `[1, 5]` softmax | `cheesy_gland`, `contagious_ecthyma`, `healthy`, `lice_infestation`, `mange_scabies` | EfficientNetB0 | 91.2% |
| Cow | `[1, 4]` softmax | `Healthy`, `Lumpy_skin`, `Mange`, `Ringworm` | EfficientNetB0 | 88.2% |

- The order above is the order in each labels file. **Wrong order means wrong disease names on the right numbers.** Always load the labels from the file shipped with the model, never retype them.
- Labels are not consistent between animals (cat uses `Health`, others use `Healthy`; capitalisation differs). Map them to **display names** in the app (for example `contagious_ecthyma` becomes "Contagious ecthyma (orf)") instead of showing raw labels.
- `mange_scabies` in the goat model is one merged class (scabies is a subtype of mange).

### Confidence rule ("unsure")

| Model | Rule | Status |
|---|---|---|
| Validator | `>= 0.5` valid animal | Tested |
| Cow | top probability **below 0.8 → "unsure"** | Tested: answers 73/93 test photos at 95.9% accuracy |
| Cat | top probability **below 0.5 → "unsure"** | Tested: answers 80.8% of test photos at 88.6% accuracy |
| Dog | starting value **0.5** | **Not yet tuned.** Placeholder until validated |
| Goat | starting value **0.5** | **Not yet tuned.** Placeholder until validated |

Keep each threshold in one config place per model. A threshold was picked for **its** model. A new model version needs a new threshold.

When the result is "unsure", **never show the best guess to the user.** Log it for debugging only.

---

## 4. Preprocessing, step by step

Do these in this exact order:

1. **Make the photo upright first.** Gallery photos: read the EXIF orientation (`ExifInterface`). CameraX: use `imageInfo.rotationDegrees`. Skipping this is the most common bug (portrait photos fail, landscape ones work).
2. **RGB only.** Drop the alpha channel. Never feed BGR.
3. **Resize the whole photo to 224 x 224, bilinear.** Stretch it. No crop, no padding.
4. **Convert to `float32`, keep values 0 to 255.** No `/255`, no `(x - mean) / std`.
5. **Do not sharpen, filter or auto-enhance** before the model.
6. Run the interpreter, then apply the rules for that model (sigmoid threshold for the validator, argmax plus threshold for Stage 2).

---

## 5. Android integration (Java, Android Studio)

The full step-by-step walkthrough (project setup, manifest, layout, every class, testing) is in **`ANDROID_INTEGRATION_GUIDE.md`**. This section is the short reference version.

### Setup

Put the 5 `.tflite` files and the 4 labels files in `app/src/main/assets/` (one flat folder). Create `cow_labels.txt` with the four cow classes, one per line, in the order from section 3. Stop Gradle from compressing the models, otherwise they cannot be memory-mapped and the app crashes on load:

```groovy
// app/build.gradle (Groovy)
android {
    aaptOptions { noCompress "tflite" }
    compileOptions {
        sourceCompatibility JavaVersion.VERSION_11
        targetCompatibility JavaVersion.VERSION_11
    }
}
dependencies {
    implementation 'com.google.ai.edge.litert:litert:2.1.0'      // check the LiteRT docs for the latest version
    implementation 'androidx.exifinterface:exifinterface:1.3.7'  // photo rotation
}
// build.gradle.kts users: androidResources { noCompress += "tflite" } and implementation("...") with parentheses
```

The `Interpreter` import depends on the artifact: `com.google.ai.edge.litert.Interpreter` (LiteRT) or `org.tensorflow.lite.Interpreter` (older TensorFlow Lite). Only that one import line changes.

### Run one model (same class for all five)

```java
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import com.google.ai.edge.litert.Interpreter;   // older artifact: org.tensorflow.lite.Interpreter
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class ModelRunner implements AutoCloseable {
    private static final int SIZE = 224;
    private final Interpreter interpreter;
    private final ByteBuffer input =
            ByteBuffer.allocateDirect(4 * SIZE * SIZE * 3).order(ByteOrder.nativeOrder());
    private final int[] pixels = new int[SIZE * SIZE];
    public final int outputSize;

    public ModelRunner(Context ctx, String modelFile) throws IOException {
        try (AssetFileDescriptor fd = ctx.getAssets().openFd(modelFile);
             FileInputStream is = new FileInputStream(fd.getFileDescriptor());
             FileChannel channel = is.getChannel()) {
            MappedByteBuffer model = channel.map(
                    FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);
            interpreter = new Interpreter(model, options);
        }
        outputSize = interpreter.getOutputTensor(0).shape()[1];
    }

    /** The bitmap must already be upright (rotation applied). Call off the main thread. */
    public synchronized float[] run(Bitmap upright) {
        Bitmap resized = Bitmap.createScaledBitmap(upright, SIZE, SIZE, true); // stretch, no crop/pad
        resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE);
        if (resized != upright) resized.recycle();

        input.rewind();
        for (int p : pixels) {
            input.putFloat((p >> 16) & 0xFF);   // R, raw 0..255
            input.putFloat((p >> 8) & 0xFF);    // G
            input.putFloat(p & 0xFF);           // B
        }
        input.rewind();

        float[][] output = new float[1][outputSize];
        interpreter.run(input, output);
        return output[0];
    }

    @Override public void close() { interpreter.close(); }
}
```

### Models, labels and thresholds in one enum

```java
public enum Animal {
    CAT ("Cat",  "cat_skin_disease_model.tflite", "cat_skin_disease_labels.txt",  0.50f),
    DOG ("Dog",  "dog_v2_float16.tflite",         "dog_labels.txt",               0.50f), // placeholder, not tuned
    GOAT("Goat", "goat_skin_disease_model.tflite","goat_skin_disease_labels.txt", 0.50f), // placeholder, not tuned
    COW ("Cow",  "cow_model_v1.tflite",           "cow_labels.txt",               0.80f);

    public final String displayName, modelFile, labelsFile;
    public final float unsureBelow;

    Animal(String displayName, String modelFile, String labelsFile, float unsureBelow) {
        this.displayName = displayName; this.modelFile = modelFile;
        this.labelsFile = labelsFile;   this.unsureBelow = unsureBelow;
    }
    @Override public String toString() { return displayName; }
}
```

### The decision flow

```java
// validator: single sigmoid output, >= 0.5 means "valid animal"
float validProb = validator.run(bitmap)[0];
if (validProb < 0.5f) return Outcome.notAnAnimal();            // "Please upload a valid animal image."

// Stage 2: only the model of the animal the user picked
float[] probs = stage2Model.run(bitmap);
int best = 0;
for (int i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;

String label = labels.get(best);                               // from the labels file, never hardcoded
if (probs[best] < animal.unsureBelow) return Outcome.unsure(label, probs[best]);   // log only, never show
return Outcome.result(label, probs[best]);
```

Notes:

- **Threading:** run `ModelRunner.run(...)` on a single background thread (for example `Executors.newSingleThreadExecutor()`), then update the UI with `runOnUiThread(...)`. Never run a model on the main thread.
- **Load once, reuse:** create each `ModelRunner` once (the validator at app start, each Stage-2 model the first time its animal is used) and call `close()` in `onDestroy`. Do not reload a model for every photo.
- **Labels check:** after loading a model, compare the number of lines in its labels file with `outputSize`. A mismatch means the wrong labels file and must fail loudly.
- **Rotation:** use `androidx.exifinterface.media.ExifInterface` to read the orientation, then rotate the bitmap upright **before** `run(...)`. The guide has a ready `ImageUtils.loadUpright(...)`.
- The models are float16-weight exports. The input and output tensors are still `float32`.
- Do **not** swap in an 8-bit (dynamic range) export. Cat and cow were tested with it and it changed predicted classes on several test photos. Use float32 or float16 only.

---

## 6. What the app must show the user

1. **Below the threshold, show "unsure".** Wording: *"We couldn't tell from this photo. Try another photo in good light, or consult a veterinarian."*
2. **Always show:** *"This is a screening aid, not a veterinary diagnosis."*
3. **Never show a bare "Healthy".** The models only know a few skin conditions. Wording: *"No signs of [the model's diseases] detected. This does not rule out other conditions."*
4. **Skin diseases only.** The project scope is skin conditions. Do not present results for eye, ear, hoof, udder or other problems. The models cannot recognise them.
5. **Capture-screen guidance:** the affected skin area, in focus, in daylight, close enough to see the skin, not blurry or dark.
6. **Validator rejection message:** *"Please upload a valid animal image."*
7. **Ask the user to pick the animal first,** and show that choice again on the result screen so a wrong pick is easy to notice.

---

## 7. Model performance and honest limits

Numbers are from held-out test photos. Datasets are web images (Roboflow, search results), not farmers' phone photos, so real-world accuracy may be lower.

### Validator (Stage 1)

MobileNetV2, trained on 2,132 images (1,242 valid animal, 890 negative incl. 540 hard negatives). Test (214 images): accuracy 98.60%, precision 99.19%, recall 98.40%.

- Aspect ratio is **not** a failure mode: tall and wide images were stress-tested, direct resize is fine.
- **Memes and text-overlay images** can pass as "valid animal".
- Other real animals (not cat, dog, cow, goat) can pass as "valid animal".

### Cat (weakest model)

EfficientNetB0, 224 px, float16 TFLite. Test accuracy **78.46%**; at the 0.5 cut-off it answers 80.8% of photos at 88.6% accuracy. Ringworm is the weakest class (about 77 to 78% recall).

- On an independent cat dataset (904 unseen photos) the shipped model scored only **27.7%**. The cat classes (especially ringworm, dermatitis and scabies) are visually similar and labels differ between datasets. Treat cat results with extra caution and keep the "unsure" rule on.
- Adding the extra datasets did not help in cross-validation, so more data cleaning or expert-labelled data is needed to improve it.

### Dog

MobileNetV2, float16 TFLite. Test accuracy **89.1%** on 5 classes (Dermatitis, Healthy, Hypersensitivity, demodicosis, ringworm). The Fungal_infections class was dropped because it was the weakest. Confidence threshold not yet tuned.

### Goat

EfficientNetB0, float16 TFLite. Test accuracy **91.2%** on 5 skin-only classes. Known limits:

- `contagious_ecthyma` recall is about 79% and has very few unique real source photos (the classes `lice_infestation` also has a thin real-photo pool, mostly augmented).
- Scabies and mange could not be separated reliably, so they are one merged class.
- Pink eye was removed because it is an eye condition (out of scope).

### Cow

EfficientNetB0, 224 px, 3 skin diseases + Healthy. Test accuracy **88.2%** (82 of 93). At the 0.8 rule it answers 73 of 93 photos at 95.9% accuracy, and 0 of 33 sick test photos were called Healthy.

- The test set is small. Mange has only 6 test photos, so one photo moves its recall by about 17 points.
- Lumpy skin and Mange get confused with each other. Mange is the weakest class.
- Some wrong answers were confident (0.83 to 0.99). The 0.8 rule reduces mistakes but cannot remove them.
- See `models/cow/cow_model_v1_deployment_guide.md` for the full cow hand-off notes.

---

## 8. Verify before every release

The app's preprocessing is the most likely place for a bug. Compare the phone against this reference script on a computer (`pip install ai-edge-litert pillow numpy`, or `tensorflow pillow`).

```python
import sys, numpy as np
from PIL import Image, ImageOps
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

MODEL = sys.argv[1]                      # e.g. models/cow/cow_model_v1.tflite
CLASSES = sys.argv[2].split(",")         # e.g. Healthy,Lumpy_skin,Mange,Ringworm
THRESHOLD = float(sys.argv[3])           # e.g. 0.8

it = Interpreter(model_path=MODEL); it.allocate_tensors()
inp, out = it.get_input_details()[0], it.get_output_details()[0]

for path in sys.argv[4:]:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.asarray(img, dtype="float32")[None]          # raw 0..255, no scaling
    it.set_tensor(inp["index"], x); it.invoke()
    p = it.get_tensor(out["index"])[0]
    verdict = "UNSURE" if p.max() < THRESHOLD else CLASSES[int(p.argmax())]
    print(path, dict(zip(CLASSES, p.round(3))), "->", verdict)
```

Example:

```
python check.py models/cow/cow_model_v1.tflite Healthy,Lumpy_skin,Mange,Ringworm 0.8 test1.jpg test2.jpg
```

**Checklist**

- [ ] Take about 5 test photos per animal (at least one per class) and run them through both the script and the app.
- [ ] Same top class and similar probabilities. Gaps of a few percent are normal (libraries resize slightly differently). A different top class or a big gap means a preprocessing bug (section 9).
- [ ] A portrait-orientation photo gives the same result as a landscape one.
- [ ] A PNG or screenshot gives the same result as a JPEG.
- [ ] The validator rejects obvious non-animals (a car, a document, a person) and accepts clear animal photos.
- [ ] The model files in the app match the files in this repo (compare `sha256`, for example `certutil -hashfile file.tflite SHA256` on Windows).

---

## 9. Troubleshooting

| Symptom | Likely cause |
|---|---|
| Almost every photo returns the same class, or confidence is low everywhere | Pixels were divided by 255 or normalised, or channels are BGR |
| Portrait photos give wrong results, landscape ones are fine | Rotation not applied before resizing |
| Result differs a lot from the Python script | Crop or padding used instead of stretch, or wrong pixel range |
| App crashes on model load | Model compressed in the APK (`noCompress "tflite"` missing) or wrong file name or path |
| Disease names do not match the photo | Class order does not match the labels file |
| Validator rejects real animal photos | Double preprocessing, or an extra normalisation step in the app |
| Much higher "unsure" rate than expected | Photos too blurry, dark, far away, or of the wrong body area |
| Predictions changed after re-export | Model was quantized to 8-bit. Re-export as float32 or float16 and re-run the agreement check |

---

## 10. Keeping the models reliable

### Never change these without re-testing

| Do not change | Why |
|---|---|
| Preprocessing (section 4) | Silent wrong answers |
| Class order | Wrong disease names |
| Model file | A different export changes the results |
| Confidence thresholds | Each was picked for its own model |
| Quantization | Use float32 or float16 only, never 8-bit |

If anyone re-exports a model, compare it with its `.keras` file on the frozen test set before shipping. Accept it only if predicted classes agree on at least 99% of photos and the confidence gap is small (float16 gave about 0.006 for the cow model).

### Versioning

- Name files `..._v1.tflite`, `v2`, and so on. **Never overwrite a version.**
- Keep each model and its labels (and threshold) together as a pair.
- Record the SHA-256 of every shipped model in the release notes, with a short changelog (what changed, why, test numbers).

### In the field

- Count how often the app says "unsure". If real use is far above the test rate (cow: about 22%), users are taking photos unlike the training data.
- Add an in-app "Was this wrong?" button. Log wrong or unsure photos **only with user consent**. They are the best data for the next version.

### Improvement backlog (in order of value)

1. Real phone photos from farms and clinics, especially for cat, cow mange, and goat ecthyma and lice.
2. Veterinarian review of doubtful labels.
3. Tune the confidence thresholds for dog and goat on validation data.
4. Field validation on real photos before making accuracy claims in the final report.

### Retraining lessons (from this project)

- One copy per source photo. Roboflow exports contain several augmented copies of the same photo.
- Remove near-duplicates by image similarity, not file bytes, and **split by group** so look-alike photos never cross train/val/test.
- Keep non-skin conditions (eye, hoof, udder) out of skin models.
- Remove diagrams, screenshots, logos and photos with captions naming a disease.
- Same augmentation for every class, training set only. Never tune on the test set.

---

## 11. Stage-1 validator details

- **Architecture:** MobileNetV2 (ImageNet pretrained) + GlobalAveragePooling2D → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → Dense(1, sigmoid). MobileNetV2 `preprocess_input` is built in as an internal layer.
- **Input:** `[1, 224, 224, 3]` float32, raw 0 to 255 (see section 4).
- **Output:** one sigmoid value. `>= 0.5` valid animal, `< 0.5` negative. Class order `0 = negative`, `1 = valid_animal`.
- **Training notebook:** `notebooks/validator.ipynb` (dataset prep, training, evaluation, TFLite conversion, stress testing).

| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.978 | 0.989 | 0.983 | 89 |
| Valid Animal | 0.992 | 0.984 | 0.988 | 125 |

---

## Team

- Abdullah Khan, Team Lead (ML models)
- Ghazala Sultana
- Alia Rubab
- Alishba Noor

Supervisor: Ms. Rameeza Shaheen
