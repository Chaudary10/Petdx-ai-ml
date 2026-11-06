# PETDX AI: Cow Skin Disease Model v1

**Deployment guide and maintenance rules (hand-off to the app team)**
Trained September 2026. Cattle only.

The model will never be "perfect". What this guide protects is the behaviour we **measured**: if the app feeds the model exactly what it was trained on, and nobody changes the pieces listed in section 8, the app performs as reported in section 7.

---

## 1. Files in the hand-off

| File | What it is | Ship in app? |
|---|---|---|
| `cow_model_v1.tflite` | The model (8.1 MB, float16 weights) | **Yes** |
| `cow_config.json` | Class order, input size, confidence rule | **Yes** (must travel with the model) |
| `cow_model.keras` | Trained model, used only to re-export | No |
| `cow_split.zip` | Frozen cleaned dataset and test set, used to re-check any new model | No |

**Use only `cow_model_v1.tflite`.** An earlier 8-bit version (`cow_model.tflite`, 4.6 MB) was tested and rejected: it changed the predicted class on 4 of 93 test photos and shifted confidences by up to 0.25. If you find that file anywhere, delete it.

---

## 2. Model contract

| Item | Value |
|---|---|
| Input | one image, shape `[1, 224, 224, 3]`, `float32`, RGB |
| Pixel values | **raw 0 to 255. Do NOT divide by 255.** The model scales internally |
| Output | shape `[1, 4]`, `float32`, probabilities that sum to 1 |
| Class order | `0 Healthy`, `1 Lumpy_skin`, `2 Mange`, `3 Ringworm` |
| Decision rule | if the highest probability is **below 0.8**, show "unsure". Otherwise show that class |

---

## 3. Preprocessing (must match training exactly)

1. **Apply the photo's rotation first.** Gallery photos: read EXIF orientation (`ExifInterface`). CameraX: use `imageInfo.rotationDegrees`.
2. **RGB only.** Drop the alpha channel. Do not feed BGR.
3. **Resize the whole photo to 224 x 224, bilinear.** Stretch it. **No center-crop, no padding.** Training used the same stretch.
4. **Keep values 0 to 255 as float32.** No `/255`, no mean/std normalisation.
5. **Do not sharpen, filter, or auto-enhance** before the model.

Getting any of these wrong does not crash the app. It silently gives wrong answers and looks like a broken model.

---

## 4. Android integration (Kotlin)

Put both files in `app/src/main/assets/`. Stop Gradle from compressing the model, otherwise it cannot be memory-mapped:

```kotlin
// app/build.gradle.kts
android {
    androidResources { noCompress += "tflite" }   // older Gradle: aaptOptions { noCompress "tflite" }
}
```

Add the LiteRT (formerly TensorFlow Lite) dependency. Check the official LiteRT docs for the current artifact and version. The `Interpreter` import differs by artifact:
`org.tensorflow.lite.Interpreter` (TensorFlow Lite) or `com.google.ai.edge.litert.Interpreter` (LiteRT).

```kotlin
import android.content.Context
import android.graphics.Bitmap
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import org.tensorflow.lite.Interpreter

sealed class CowResult {
    data class Label(val name: String, val confidence: Float) : CowResult()
    // bestGuess is for logging only. Do NOT show it to the user.
    data class Unsure(val bestGuess: String, val confidence: Float) : CowResult()
}

class CowClassifier(context: Context) {
    // Must match cow_config.json. Order matters.
    private val classes = listOf("Healthy", "Lumpy_skin", "Mange", "Ringworm")
    private val unsureBelow = 0.8f
    private val size = 224
    private val interpreter: Interpreter

    init {
        val fd = context.assets.openFd("cow_model_v1.tflite")
        val model = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        interpreter = Interpreter(model)
    }

    /** The bitmap must already be upright (rotation applied). Call off the main thread. */
    fun classify(bitmap: Bitmap): CowResult {
        val resized = Bitmap.createScaledBitmap(bitmap, size, size, true)  // stretch, no crop/pad
        val pixels = IntArray(size * size)
        resized.getPixels(pixels, 0, size, 0, 0, size, size)

        val input = ByteBuffer.allocateDirect(4 * size * size * 3).order(ByteOrder.nativeOrder())
        for (p in pixels) {
            input.putFloat(((p shr 16) and 0xFF).toFloat())  // R, raw 0..255
            input.putFloat(((p shr 8) and 0xFF).toFloat())   // G
            input.putFloat((p and 0xFF).toFloat())           // B
        }
        input.rewind()

        val output = Array(1) { FloatArray(classes.size) }
        interpreter.run(input, output)

        val probs = output[0]
        val best = probs.indices.maxByOrNull { probs[it] } ?: 0
        return if (probs[best] < unsureBelow) CowResult.Unsure(classes[best], probs[best])
        else CowResult.Label(classes[best], probs[best])
    }

    fun close() = interpreter.close()
}
```

---

## 5. Verify before every release

The app's preprocessing is the most likely place for a bug. Compare the phone against a reference run on a computer.

**Reference script** (`pip install tensorflow pillow`):

```python
import sys, numpy as np, tensorflow as tf
from PIL import Image, ImageOps

CLASSES = ["Healthy", "Lumpy_skin", "Mange", "Ringworm"]
it = tf.lite.Interpreter(model_path="cow_model_v1.tflite"); it.allocate_tensors()
inp, out = it.get_input_details()[0], it.get_output_details()[0]

for path in sys.argv[1:]:
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.asarray(img, dtype="float32")[None]           # 0..255, no scaling
    it.set_tensor(inp["index"], x); it.invoke()
    p = it.get_tensor(out["index"])[0]
    print(path, dict(zip(CLASSES, p.round(3))), "->", "UNSURE" if p.max() < 0.8 else CLASSES[p.argmax()])
```

**Checklist**

- [ ] Take about 5 photos from `cow_split.zip` (`test/` folders, at least one per class) and run them through both the script and the app.
- [ ] Same top class and similar probabilities. Small gaps (a few percent) are normal because libraries resize slightly differently. A different top class or a large gap means a preprocessing bug (see section 9).
- [ ] Portrait-orientation photos give the same result as landscape ones (rotation check).
- [ ] A photo sent to the app as PNG/screenshot works the same as a JPEG.
- [ ] `sha256` of the shipped model matches the value recorded in the release notes:
  Windows `certutil -hashfile cow_model_v1.tflite SHA256`, Linux/Mac `sha256sum cow_model_v1.tflite`.

---

## 6. How the app must present results

**Rules**

1. **Below 0.8 show "unsure".** Never show the best guess. Wording: *"We couldn't tell from this photo. Try another photo in good light, or consult a veterinarian."*
2. **Present results as a screening aid, not a diagnosis.** Always show: *"This is not a veterinary diagnosis."*
3. **Never say "healthy" without qualifying it.** The model only knows three diseases. Wording: *"No signs of lumpy skin disease, mange or ringworm detected. This does not rule out other conditions."*
4. **Cattle only.** Route by the animal the user picks. Never run a cow photo through another animal's model, or the reverse.
5. **Photo guidance on the capture screen:** the affected skin area, in focus, in daylight, close enough to see the skin, not blurry or dark.
6. **Closed set:** the model has no "other" class. Anything outside the four classes (udder or mastitis photos, hoof, eye or mouth problems, other diseases, non-cattle, non-animals) will still be forced into one of them, sometimes with high confidence. The photo guidance and the disclaimers are the only protection.

---

## 7. What the model can and cannot do (measured)

**Setup:** EfficientNetB0 (ImageNet pretrained), 224 x 224, 3 skin diseases + Healthy. Trained on 435 photos, validated on 93, tested once on 93 held-out photos.

| Class | Train | Val | Test |
|---|---|---|---|
| Healthy | 280 | 60 | 60 |
| Lumpy skin | 62 | 12 | 13 |
| Mange | 26 | 6 | 6 |
| Ringworm | 67 | 15 | 14 |

**Results on the test set**

| | Value |
|---|---|
| Accuracy with no cut-off | 88% (82 of 93) |
| At the 0.8 rule: photos answered | 73 of 93 (78%) |
| At the 0.8 rule: accuracy on answered photos | 95.9% (3 wrong of 73) |
| Sick cows called Healthy (answered) | 0 of 33 sick test photos |
| "Unsure" rate | about 22% of all photos, about 33% of sick-cow photos |

| Class | Recall (no cut-off) | Answered at 0.8 |
|---|---|---|
| Healthy | 93% (56/60) | 51/60 |
| Lumpy skin | 77% (10/13) | 8/13 |
| Mange | 67% (4/6) | 3/6 |
| Ringworm | 86% (12/14) | 11/14 |

**Limits to state honestly**

- **The test set is small.** Mange has only 6 test photos, and one photo moves its recall by about 17 points. "0 sick called Healthy" means 0 of 33 photos, not "never".
- **Lumpy skin and Mange get confused with each other.** This is the main error pattern.
- **The data comes from web images** (search results, Roboflow, screenshots), not from farmers' phones. Real-world accuracy may be lower until it is checked on real farm photos.
- **A few labels are doubtful** (about 3 test/val photos). If they are wrong, the true accuracy is slightly higher.
- **The 0.8 rule reduces mistakes but cannot remove them.** Some wrong answers were confident (0.83 to 0.99).

---

## 8. Keeping the model reliable

### Never change these without re-testing

| Do not change | Why |
|---|---|
| Preprocessing (section 3) | Silent wrong answers |
| Class order | Wrong disease names on the right numbers |
| Model file | Swapping in a different export changes the results |
| The 0.8 threshold | It was picked on validation data for **this** model. A new model needs a new threshold |
| Do NOT use 8-bit ("dynamic range") quantization | Tested: 95.7% agreement, 0.25 confidence shift. Use float32 or float16 only |

If anyone re-exports the model, re-check it against `cow_model.keras` on the frozen test set before shipping. Accept it only if: predicted classes agree on at least 99% of photos, the maximum confidence gap is small (float16 gave 0.006), and the count answered at 0.8 and wrong answers are unchanged.

### Versioning

- Name files `cow_model_v1.tflite`, `v2`, and so on. **Never overwrite a version.**
- Ship `cow_config.json` with the model. The model and its threshold are a pair.
- Keep the SHA-256 of each shipped model in the release notes.
- Keep a short changelog: what changed, why, the test numbers.

### Watch it in the field

- Count how often the app says "unsure". The test set gave about 22%. If real use shows far more (say double), users are taking photos unlike the training data.
- Add an in-app "Was this wrong?" button, and log wrong or unsure photos **only with user consent**. These are the most valuable data for the next version.
- Review a sample of logged photos regularly. New failure patterns show up there first.

### Retraining checklist (lessons from this training run)

Most of the work on v1 was cleaning data. These mistakes were found and fixed. Do not reintroduce them:

1. **One copy per source photo.** Roboflow exports contain several augmented copies of the same photo (filenames with `.rf.<hash>`).
2. **Remove near-duplicates by image similarity, not by file bytes.** The same photo appeared in different crops, colours, and rotations, and once under two different labels.
3. **Split by group.** Look-alike photos (similarity 0.90 or higher) must stay in the same split, otherwise the test score is inflated.
4. **Keep non-skin conditions out** (udder or mastitis, hoof, eye). Two mastitis screenshots were mislabelled as lumpy skin.
5. **Remove diagrams, drawings, web-page screenshots, logos and watermarks,** and photos that carry a caption naming a disease (the model learns the text instead of the skin).
6. **Check labels against filenames and captions.** Several photos carried a name for a different disease than their folder.
7. **Same augmentation for every class, training set only.** Fill rotated corners by reflecting the edge, not with black.
8. **Never tune on the test set.** Pick thresholds and settings on validation, then test once.
9. **Never add test photos to training.** `cow_split.zip`'s test set stays frozen so versions can be compared fairly.

### Improvement backlog (in order of value)

1. **More Mange and Lumpy skin photos**, especially real phone photos from farms. Mange has only 26 training photos.
2. **Have a supervisor or veterinarian review the doubtful labels.** A few photos are flagged as possibly mislabelled.
3. **A 4th cow skin disease,** if the supervisor wants one.
4. **Field validation** on real farm photos before making accuracy claims in the report.

---

## 9. Troubleshooting

| Symptom | Likely cause |
|---|---|
| Nearly every photo returns the same class, or confidence is low everywhere | Pixels were divided by 255, or channels are BGR |
| Portrait phone photos give wrong results, landscape ones are fine | Rotation not applied before resizing |
| Result differs a lot from the Python script | Different resize (crop or pad instead of stretch), or wrong pixel range |
| App crashes on model load | Model is compressed in the APK (`noCompress "tflite"` missing) or the file name/path is wrong |
| Names are wrong (e.g. "Mange" for a ringworm photo) | Class order does not match `cow_config.json` |
| Much higher "unsure" rate than expected | Photos are too blurry, dark, far away, or of the wrong body area |
| Predictions changed after re-export | Model was quantized to 8-bit. Re-export as float32 or float16 and re-run the agreement check |
