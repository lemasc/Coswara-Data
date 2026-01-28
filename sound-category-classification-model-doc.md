# Sound Category Classification (9-class) — Training Documentation (Coswara paper)

This document describes how to reproduce the **sound-category classification** experiment described in the Coswara paper extract. The goal is to train a model that predicts which of the **9 recording prompts** an audio clip belongs to (breathing/cough/vowel/speech variants), using a classical ML pipeline (feature extraction → Random Forest).

---

## 1) Project overview

### Objective

Build a **9-class audio classifier** that assigns each recording to one of the Coswara sound categories (two breathing, two cough, three vowels, two speech/counting). The paper uses this task as a “sanity check” that the categories are acoustically distinct.

### Sound categories (labels)

The dataset contains the following nine categories (names may differ slightly depending on your local file naming—what matters is the mapping to these concepts): breathing-shallow, breathing-deep, cough-shallow, cough-heavy, vowel-[u], vowel-[i], vowel-[æ], counting-normal, counting-fast. 

### Data scale (context)

Coswara contains ~65 hours of audio, about 23,700 recordings from 2,635 subjects (9 recordings per subject). 

### Approach in the paper

For each audio file:

1. Compute a **short-time mel-spectrogram** using:

   * window length **25 ms**
   * hop/shift **10 ms**
   * **128 mel filters**
2. Average the mel-spectrogram over time → a **128-D vector** (“averaged mel-spectrum vector”).
3. Train a **Random Forest** classifier (criterion: **gini**) with a **70/15/15** stratified train/val/test split, tuning the **number of trees** (estimators) on the validation set. 

---

## 2) Repository & folder expectations

You said all dataset files are in `dataset/`. A practical structure (you can adapt to whatever you actually have) is:

```
dataset/
  audio/                      # wav/flac files (or nested subject folders)
  metadata.csv                # optional (recommended)
  quality.csv                 # optional (if you have human quality labels)
```

### Recommended: build a manifest

Create a single table (CSV/Parquet) with one row per audio file:

| column       | description                                         |
| ------------ | --------------------------------------------------- |
| `path`       | relative path to audio file                         |
| `label`      | one of the 9 categories                             |
| `subject_id` | optional but recommended for leakage-safe splitting |
| `quality`    | optional (0/1/2 if available)                       |
| `sr`         | optional (detected sample rate)                     |
| `duration_s` | optional                                            |

How you populate `label` depends on your local file naming. Common options:

* infer from filename prefix/suffix
* infer from parent directory name
* read from an existing metadata file

---

## 3) Environment setup (Python)

### Suggested dependencies

* `numpy`, `pandas`
* `librosa` (or `torchaudio`) for audio + mel features
* `scikit-learn` for RandomForest + split + metrics
* `matplotlib` (optional) for plotting confusion matrix

Example install:

```bash
pip install numpy pandas librosa scikit-learn matplotlib soundfile
```

---

## 4) Data preparation

### 4.1 Audio loading

Guidelines:

* Load mono (convert stereo → mono).
* Keep native sampling rate or resample consistently. The paper doesn’t specify resampling; for reproducibility, pick one strategy and keep it fixed (e.g., `sr=None` in librosa to keep original SR, or resample everything to 48 kHz).

### 4.2 Optional: filter by quality

If you have the paper’s manual quality ratings (excellent/moderate/poor), you may:

* **include all files** for the sound-category task (the paper’s sound-category section doesn’t explicitly say it filtered)
* or **exclude poor quality (rating 0)** to reduce noise (often improves accuracy)

---

## 5) Feature extraction (match the paper)

### 5.1 Mel-spectrogram parameters

Use:

* window = **25 ms**
* hop = **10 ms**
* n_mels = **128**

In sample counts:

* `n_fft` / `win_length` ≈ `0.025 * sr`
* `hop_length` ≈ `0.010 * sr`

Then average over time frames to get a fixed-length vector:

* `feature = mean(mel_spectrogram, axis=time)  → shape (128,)`

This is exactly what the paper describes. 

### 5.2 Practical implementation notes

* Use **power mel-spectrogram** (default in many pipelines). If you use log-mel, document it and keep it consistent.
* Consider adding a small epsilon before log if you do log compression.
* Handle variable-length recordings naturally: averaging removes the time dimension.

### 5.3 Pseudocode (developer-facing, not full implementation)

```python
def extract_feature(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    win_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)

    # Choose n_fft >= win_length, typically power of 2
    n_fft = 1
    while n_fft < win_length:
        n_fft *= 2

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=128,
        power=2.0,
    )

    feat = mel.mean(axis=1)  # (128,)
    return feat
```

---

## 6) Dataset splitting (train/val/test)

### Paper split

Use a stratified random sampling split: **70% train / 15% val / 15% test**. 

### Important engineering decision: split by subject vs by file

* **File-level stratified split** (likely closest to “random sampling” as written) is easiest.
* **Subject-level split** is safer against leakage (same person’s microphone/environment/voice characteristics appearing in train and test). This can reduce accuracy but is more realistic.

Recommended compromise:

* implement **subject-level split** if `subject_id` is available
* otherwise do file-level but clearly document that leakage may inflate results

---

## 7) Model training (Random Forest)

### 7.1 Model choice

The paper uses a **RandomForestClassifier** with:

* `criterion="gini"`
* tune **`n_estimators`** on the validation set 

### 7.2 Tuning strategy (simple + reproducible)

1. Fit RF with a small grid of `n_estimators`, e.g. `[100, 200, 500, 1000]`
2. Choose the best by **validation accuracy**
3. Retrain on **train + val** (optional) using best `n_estimators`
4. Evaluate once on test

Pseudocode sketch:

```python
candidates = [100, 200, 500, 1000]
best = None

for n in candidates:
    clf = RandomForestClassifier(
        n_estimators=n,
        criterion="gini",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    update_best(...)
```

---

## 8) Model validation & evaluation

### 8.1 Primary metric

* **Accuracy** on the test set (paper reports overall accuracy)

### 8.2 Additional metrics (recommended)

* Confusion matrix (paper reports this and interprets errors)
* Per-class accuracy (reveals which prompts are hardest)
* Macro-F1 (helpful if classes are slightly imbalanced)

### 8.3 Confusion matrix interpretation

Expect most confusions to be “within-family”:

* breathing-deep vs breathing-shallow
* cough-heavy vs cough-shallow
* counting-fast vs counting-normal
  The paper notes a “block diagonal” structure reflecting these grouped confusions. 

---

## 9) Expected results (per the authors)

If you follow the paper’s described setup (128-D averaged mel vector + RF + 70/15/15 stratified split + tuning estimators on val), the paper reports:

* **Test accuracy: 56.5%**
* This is well above **chance (11.1%)** for 9 classes
* Confusions mostly occur within broad groups (breathing vs cough vs speech). 

Note: Your reproduced accuracy can vary due to:

* different split randomness (set seeds!)
* whether you split by subject (often lower but more realistic)
* whether you filter poor-quality audio
* differences in preprocessing (resampling, log-mel vs power mel, silence trimming)

---

## 10) Deliverables checklist (what a developer should produce)

1. **`manifest.csv`** creation script (paths + labels + optional subject_id/quality)
2. **Feature extraction module**

   * deterministic settings (window/hop/n_mels)
   * caching features to disk (highly recommended)
3. **Training script**

   * split generation (seeded)
   * RF training + `n_estimators` tuning
4. **Evaluation report**

   * test accuracy
   * confusion matrix plot + brief analysis
   * saved model artifact (`joblib`/`pickle`)
5. **Reproducibility**

   * fixed seeds
   * logged parameters + dataset version/hash