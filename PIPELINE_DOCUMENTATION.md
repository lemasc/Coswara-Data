# Coswara Dataset Processing Pipeline Documentation

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Data Flow](#data-flow)
5. [Script Reference](#script-reference)
6. [Execution Workflow](#execution-workflow)
7. [Technical Details](#technical-details)
8. [Output Structure](#output-structure)
9. [Troubleshooting](#troubleshooting)
10. [Memory and Performance](#memory-and-performance)

---

## Overview

This pipeline transforms the raw Coswara dataset into a Hugging Face-compatible dataset with two configurations:

- **`metadata` configuration**: Participant health and demographic information (36 fields)
- **`audio` configuration**: Individual audio recordings with quality annotations and denormalized metadata (11 fields)

The pipeline is designed for memory-efficient processing of 2,746 participants with 24,718 audio files (~9 recordings per participant).

### Design Principles

- **Batch processing**: Process data in small batches (default: 50 participants) to manage memory
- **Resumability**: Each batch saved independently, allowing restart from any point
- **Validation**: Comprehensive checks at each phase
- **Parallel processing**: Multi-core audio processing support
- **Test mode**: Ability to test with subset of data before full processing

### Final Dataset

- **Repository**: Uploaded to Hugging Face
- **Format**: Multi-file Parquet (native HuggingFace support)
- **Splits**: train (70%), val (15%), test (15%) - stratified by COVID status
- **Audio**: 16kHz WAV (resampled from mixed 48kHz/44.1kHz/16kHz sources)

---

## Prerequisites

### Required Data Files

Before running the pipeline, ensure these files/folders exist in the repository root:

```
Coswara-Data/
├── combined_data.csv           # Metadata for all 2,746 participants
├── csv_labels_legend.json      # Field definitions
├── annotations/                # Quality labels (0=bad, 1=good, 2=excellent)
│   ├── breathing-deep_labels.csv
│   ├── breathing-shallow_labels.csv
│   ├── cough-heavy_labels.csv
│   ├── cough-shallow_labels.csv
│   ├── vowel-a_labels.csv
│   ├── vowel-e_labels.csv
│   ├── vowel-o_labels.csv
│   ├── counting-normal_labels.csv
│   └── counting-fast_labels.csv
└── Extracted_data/             # Extracted audio files (from extract_data.py)
    ├── 20200413/
    │   ├── {participant_id}/
    │   │   ├── breathing-deep.wav
    │   │   ├── breathing-shallow.wav
    │   │   ├── cough-heavy.wav
    │   │   ├── cough-shallow.wav
    │   │   ├── vowel-a.wav
    │   │   ├── vowel-e.wav
    │   │   ├── vowel-o.wav
    │   │   ├── counting-normal.wav
    │   │   └── counting-fast.wav
    │   └── ...
    ├── 20200415/
    └── ...
```

**Note**: Compressed archive folders (`YYYYMMDD/*.tar.gz.*`) are not required for the pipeline.

### Python Dependencies

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies:

- `librosa` - Audio loading and resampling
- `datasets` - HuggingFace dataset creation
- `pandas` - Metadata processing
- `scikit-learn` - Stratified splitting
- `huggingface_hub` - Dataset uploading
- `tqdm` - Progress tracking

---

## Pipeline Architecture

The pipeline consists of 5 sequential phases:

```
Phase 1: Data Preparation
├─ Load combined_data.csv (2,746 participants)
├─ Load 9 annotation files (quality labels)
├─ Merge metadata + quality scores
├─ Build participant-to-date mapping
├─ Create stratified train/val/test splits (70/15/15)
└─ Generate batches (50 participants each)

Phase 2a: Audio Processing
├─ For each batch:
│  ├─ Load audio files (9 per participant)
│  ├─ Resample to 16kHz (handles 48kHz/44.1kHz/16kHz sources)
│  ├─ Create HuggingFace Audio features
│  ├─ Add quality scores + denormalized metadata
│  └─ Save as Parquet (audio-{split}-{batch}.parquet)

Phase 2b: Batch Validation
├─ Verify all batches exist
├─ Check file integrity
├─ Count NULL audio files
├─ Validate quality distributions
└─ Generate validation report

Phase 3: Metadata Processing
├─ Split metadata by train/val/test
└─ Save as Parquet (metadata/{split}.parquet)

Phase 4: Upload to Hugging Face
├─ Verify authentication
├─ Verify workspace integrity
├─ Upload metadata/ folder
├─ Upload audio/ folder (multi-file Parquet)
└─ Upload dataset card (README.md)
```

---

## Data Flow

### Input Data

```
combined_data.csv (2,746 rows × 36 columns)
  ├─ id: participant identifier (hash)
  ├─ a: age
  ├─ g: gender
  ├─ covid_status: 8 categories
  ├─ test_status, test_date, testType
  ├─ vacc: vaccination status
  ├─ l_c, l_s, l_l: location (country/state/locality)
  ├─ symptoms: fever, cough, bd, loss_of_smell, st, ftg, mp, diarrhoea
  ├─ comorbidities: smoker, cold, ht, diabetes, asthma, ihd, cld, pneumonia
  └─ others: rU, ep, ctScan, ctDate, ctScore, etc.

annotations/*.csv (9 files)
  ├─ FILENAME: {participant_id}_{audio_type}
  └─ QUALITY: 0 (bad) / 1 (good) / 2 (excellent)

Extracted_data/{YYYYMMDD}/{participant_id}/{audio_type}.wav
  └─ Mixed sample rates: 48kHz (majority), 44.1kHz (some), 16kHz (few)
```

### Intermediate Files (workspace/)

```
workspace/
├── metadata_with_quality.csv      # combined_data.csv + quality scores
├── splits.json                    # Participant IDs by split
├── batches.json                   # Batch assignments
├── participant_date_mapping.json  # Participant → date folder lookup
├── audio_summary.json             # Phase 2a statistics
├── validation_report.json         # Phase 2b results
├── metadata_summary.json          # Phase 3 statistics
├── metadata/                      # Final metadata Parquet files
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
└── audio/                         # Final audio Parquet files
    ├── train/
    │   ├── audio-train-00001.parquet
    │   ├── audio-train-00002.parquet
    │   └── ... (39 batches total)
    ├── val/
    │   └── ... (9 batches)
    └── test/
        └── ... (9 batches)
```

### Output Dataset (HuggingFace)

```
[username/dataset-name]
├── metadata/
│   ├── train.parquet    # 1,922 participants × 36 fields
│   ├── val.parquet      # 412 participants × 36 fields
│   └── test.parquet     # 412 participants × 36 fields
├── audio/
│   ├── train/
│   │   └── audio-train-*.parquet  # ~17,298 rows (1,922 × 9)
│   ├── val/
│   │   └── audio-val-*.parquet    # ~3,708 rows (412 × 9)
│   └── test/
│       └── audio-test-*.parquet   # ~3,708 rows (412 × 9)
└── README.md            # Dataset card
```

**Total**: ~24,718 audio recordings across 2,746 participants

---

## Script Reference

### 01_prepare_splits.py

**Purpose**: Phase 1 - Data preparation, creates stratified splits and batches

**Arguments**:

```bash
--workspace DIR       # Output directory (default: ./workspace)
--batch-size N        # Participants per batch (default: 50)
--test-mode N         # Process only first N batches per split (for testing)
--seed N              # Random seed for splitting (default: 42)
```

**What it does**:

1. Loads `combined_data.csv` (2,746 participants)
2. Loads 9 annotation files from `annotations/`
3. Merges metadata with quality scores (adds 9 quality columns)
4. Scans `Extracted_data/` to build participant → date folder mapping
5. Creates stratified 70/15/15 splits by `covid_status`
6. Divides each split into batches of 50 participants
7. Saves 4 files to workspace:
   - `splits.json` - Participant IDs by split
   - `batches.json` - Batch assignments with participant lists
   - `participant_date_mapping.json` - Date folder lookup
   - `metadata_with_quality.csv` - Full metadata with quality scores

**Example**:

```bash
# Full production run
python 01_prepare_splits.py --workspace ./workspace

# Test mode: only 2 batches per split (100 participants per split)
python 01_prepare_splits.py --workspace ./workspace --test-mode 2
```

**Output validation**:

- Prints COVID status distribution across splits (should be ~equal)
- Warns if participants missing from `Extracted_data/`
- Shows batch counts: train (~39 batches), val (~9 batches), test (~9 batches)

---

### 02_process_audio_batch.py

**Purpose**: Phase 2a - Audio processing with resampling and batch processing

**Arguments**:

```bash
--workspace DIR          # Workspace directory (default: ./workspace)
--split {train|val|test|all}  # Which split to process (default: all)
--batch ID               # Process specific batch ID (e.g., "00001")
--num-workers N          # Parallel workers (default: cpu_count - 1)
--extracted-data DIR     # Path to Extracted_data (default: ./Extracted_data)
```

**What it does**:

1. Loads configuration from workspace (batches.json, metadata, date mapping)
2. For each batch (50 participants × 9 audio types = ~450 files):
   - Parallel processing: spawns N workers to process participants
   - For each audio file:
     - Load audio at native sample rate (48kHz/44.1kHz/16kHz)
     - Resample to 16kHz if needed (using librosa)
     - Create HuggingFace Audio feature dict: `{array, sampling_rate}`
     - Add quality score and 10 denormalized metadata fields
   - Save batch as Parquet with explicit schema
3. Saves summary statistics to `audio_summary.json`

**Schema (audio configuration)**:

```python
Features({
    'participant_id': Value('string'),
    'audio_type': Value('string'),
    'audio': Audio(sampling_rate=16000),  # NULL if missing/corrupted
    'quality_score': Value('int64'),      # 0/1/2, NULL if not annotated
    'record_date': Value('string'),
    'covid_status': Value('string'),
    'age': Value('int64'),
    'gender': Value('string'),
    'test_status': Value('string'),
    'vacc': Value('string'),
    'country': Value('string')
})
```

**Example**:

```bash
# Process all splits with 8 workers
python 02_process_audio_batch.py --workspace ./workspace --num-workers 8

# Process only train split
python 02_process_audio_batch.py --workspace ./workspace --split train

# Process single batch (for testing/debugging)
python 02_process_audio_batch.py --workspace ./workspace --split train --batch 00001

# Single-threaded processing (debugging)
python 02_process_audio_batch.py --workspace ./workspace --num-workers 1
```

**Memory considerations**:

- **Per batch**: ~450 audio files (50 participants × 9 types)
- **Average audio size**: ~100KB per file (after 16kHz resampling)
- **Batch memory**: ~50-100MB per batch (varies by audio length)
- **Workers**: Each worker processes one participant at a time (~9 files)
- **Recommended**: Use default workers (cpu_count - 1) unless memory constrained

**NULL audio handling**:

- Missing files: Set to NULL, log warning
- Corrupted files: Set to NULL, log error message
- Empty files: Set to NULL
- Continues processing, does not fail
- Summary reports NULL count per batch

---

### 03_validate_batches.py

**Purpose**: Phase 2b - Validates all batches processed correctly

**Arguments**:

```bash
--workspace DIR       # Workspace directory (default: ./workspace)
```

**What it does**:

1. Loads batch configuration from `batches.json`
2. For each split (train/val/test):
   - Checks all expected batch files exist
   - Loads each Parquet file
   - Counts total rows
   - Counts NULL audio entries
   - Validates audio sample rates (should all be 16kHz)
   - Computes quality score distribution
   - Computes audio type distribution (should be ~equal, 9 types)
3. Saves validation report to `validation_report.json`

**Validation checks**:

- ✓ All batches present
- ✓ Row count matches expected (participants × 9)
- ✓ No corrupted Parquet files
- ✓ Sample rate is 16kHz for all non-NULL audio
- ⚠ Reports NULL audio percentage (expected ~1-2%)

**Example**:

```bash
python 03_validate_batches.py --workspace ./workspace
```

**Expected output**:

```
TRAIN:
  Batches: 39/39
  Rows: 17298/17298
  NULL audio: 234 (1.4%)
  Quality distribution:
    Quality 0: 2856
    Quality 1: 2094
    Quality 2: 12348

VAL:
  Batches: 9/9
  ...
```

**Failure scenarios**:

- Missing batches → Lists batch IDs, suggests re-running Phase 2a
- Row count mismatch → Indicates incomplete processing
- High NULL percentage (>5%) → Check Extracted_data integrity

---

### 04_prepare_metadata.py

**Purpose**: Phase 3 - Splits metadata by train/val/test

**Arguments**:

```bash
--workspace DIR       # Workspace directory (default: ./workspace)
```

**What it does**:

1. Loads `metadata_with_quality.csv` (2,746 rows)
2. Loads split assignments from `splits.json`
3. For each split:
   - Filters metadata to participants in split
   - Verifies all participants found
   - Saves as Parquet: `metadata/{split}.parquet`
4. Saves summary to `metadata_summary.json`

**Schema (metadata configuration)**:
All 36 original fields from `combined_data.csv` plus 9 quality score columns:

- `id`, `a` (age), `g` (gender), `covid_status`, `record_date`
- `test_status`, `test_date`, `testType`, `vacc`
- `l_c`, `l_s`, `l_l` (location)
- `ep`, `rU`
- Symptoms: `fever`, `cough`, `bd`, `loss_of_smell`, `st`, `ftg`, `mp`, `diarrhoea`
- Comorbidities: `smoker`, `cold`, `ht`, `diabetes`, `asthma`, `ihd`, `cld`, `pneumonia`, `others_resp`, `others_preexist`
- CT scan: `ctScan`, `ctDate`, `ctScore`
- Quality scores: `breathing_deep_quality`, `breathing_shallow_quality`, `cough_heavy_quality`, `cough_shallow_quality`, `vowel_a_quality`, `vowel_e_quality`, `vowel_o_quality`, `counting_normal_quality`, `counting_fast_quality`

**Example**:

```bash
python 04_prepare_metadata.py --workspace ./workspace
```

**Expected output**:

```
Total participants: 2746/2746
Columns per file: 45
  train: 1922/1922 participants
  val: 412/412 participants
  test: 412/412 participants
```

**Note**: This phase is fast (~seconds) and memory-safe (only 2,746 rows).

---

### 05_upload_to_hf.py

**Purpose**: Phase 4 - Upload to Hugging Face Hub

**Arguments**:

```bash
--workspace DIR       # Workspace directory (default: ./workspace)
--repo-id ID          # HuggingFace repo (required)
--token TOKEN         # HF API token (or use huggingface-cli login)
--dry-run             # Validate without uploading
--create-repo         # Create repo if it doesn't exist
```

**What it does**:

1. Verifies HuggingFace authentication (token or cached login)
2. Verifies workspace integrity:
   - `metadata/train.parquet`, `metadata/val.parquet`, `metadata/test.parquet`
   - `audio/train/*.parquet`, `audio/val/*.parquet`, `audio/test/*.parquet`
3. Uploads files:
   - `metadata/` folder → `metadata/` in repo
   - `audio/` folder → `audio/` in repo (multi-file Parquet, auto-detected)
   - `dataset_card.md` → `README.md` in repo
4. Creates dataset repository if `--create-repo` specified

**Authentication methods** (in order of priority):

1. `--token` argument
2. `HF_TOKEN` environment variable
3. Cached credentials from `huggingface-cli login`

**Example**:

```bash
# First-time setup: login to HuggingFace
huggingface-cli login

# Dry run: validate without uploading
python 05_upload_to_hf.py --workspace ./workspace --dry-run

# Upload (creates repo if needed)
python 05_upload_to_hf.py \
  --workspace ./workspace \
  --repo-id username/dataset-name \
  --create-repo

# Upload with explicit token
python 05_upload_to_hf.py \
  --workspace ./workspace \
  --repo-id username/dataset-name \
  --token hf_xxxxxxxxxxxxx
```

**Expected output**:

```
✓ Authenticated as: lemasc
✓ Workspace verified
  Metadata files: 3
  Audio parquet files: 57

[1/3] Uploading metadata configuration...
  ✓ Uploaded metadata configuration

[2/3] Uploading audio configuration...
  ✓ Uploaded audio configuration

[3/3] Uploading dataset card...
  ✓ Uploaded dataset card

✓ Upload completed successfully!

Dataset available at:
  https://huggingface.co/datasets/username/dataset
```

**Multi-file Parquet handling**:

- HuggingFace automatically detects multiple Parquet files with same prefix
- `audio/train/audio-train-00001.parquet`, `audio-train-00002.parquet`, ... → single `train` split
- No manual combining required
- Supports streaming and memory-efficient loading

---

## Execution Workflow

### Full Production Run

```bash
# Step 0: Prepare environment
pip install -r requirements.txt
huggingface-cli login

# Step 1: Create splits and batches
python 01_prepare_splits.py --workspace ./workspace

# Step 2a: Process audio files (uses all CPU cores - 1)
python 02_process_audio_batch.py --workspace ./workspace

# Step 2b: Validate batches
python 03_validate_batches.py --workspace ./workspace

# Step 3: Prepare metadata
python 04_prepare_metadata.py --workspace ./workspace

# Step 4: Upload to HuggingFace
python 05_upload_to_hf.py \
  --workspace ./workspace \
  --repo-id username/dataset-name \
  --create-repo
```

---

### Test Run (Subset of Data)

Test with 2 batches per split (100 participants per split, 900 total audio files):

```bash
# Step 1: Create splits with test mode
python 01_prepare_splits.py --workspace ./workspace_test --test-mode 2

# Step 2a: Process audio files
python 02_process_audio_batch.py --workspace ./workspace_test

# Step 2b: Validate batches
python 03_validate_batches.py --workspace ./workspace_test

# Step 3: Prepare metadata
python 04_prepare_metadata.py --workspace ./workspace_test

# Step 4: Dry run upload (verify without uploading)
python 05_upload_to_hf.py --workspace ./workspace_test --dry-run
```

---

### Resume from Failure

If Phase 2a fails mid-processing, you can resume:

```bash
# Check which batches completed
ls workspace/audio/train/

# Re-run missing batches only
# Option 1: Re-run entire split (skips existing batches automatically if code modified)
python 02_process_audio_batch.py --workspace ./workspace --split train

# Option 2: Process specific batch
python 02_process_audio_batch.py --workspace ./workspace --split train --batch 00015

# After fixing all batches, continue validation
python 03_validate_batches.py --workspace ./workspace
```

**Note**: Current implementation overwrites existing batches.

---

## Technical Details

### Stratified Splitting

**Goal**: Maintain COVID status distribution across all splits

**Method**: Two-stage `train_test_split`

1. Split: (train+val) vs test (85% vs 15%)
2. Split: train vs val (70% vs 15% of original)

**COVID Status Categories** (8 classes):

- `healthy` - 52.2%
- `positive_mild` - 15.5%
- `no_resp_illness_exposed` - 9.0%
- `positive_moderate` - 6.0%
- `resp_illness_not_identified` - 5.7%
- `recovered_full` - 5.3%
- `positive_asymp` - 3.3%
- `under_validation` - 3.0%

**Verification**: Script prints distribution for each split

---

### Audio Resampling

**Source formats**:

- Majority: 48kHz WAV
- Some: 44.1kHz WAV
- Few: 16kHz WAV

**Target**: 16kHz (standard for speech/audio ML)

**Method**: `librosa.resample()`

```python
audio, sr = librosa.load(audio_path, sr=None)  # Load at native rate
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
```

**Why 16kHz?**

- Reduces file size by 67% (48kHz → 16kHz)
- Sufficient for respiratory sound analysis (human speech: 0-8kHz)
- Standard for speech models (Wav2Vec2, HuBERT, etc.)
- Nyquist frequency 8kHz captures all relevant respiratory sounds

---

### Quality Score Annotations

**Labeling process**: Manual annotation by trained annotators

**Scale**:

- **0 (bad)**: Poor quality, significant noise/distortion → Exclude from classification
- **1 (good)**: Acceptable quality for analysis
- **2 (excellent)**: High quality, clear audio

**Distribution** (example for breathing-deep):

- Quality 2: ~71.5%
- Quality 1: ~12.1%
- Quality 0: ~16.5%

**Usage**:

```python
# Filter for high-quality audio
ds = load_dataset("username/dataset-name", "audio")
ds_filtered = ds.filter(lambda x: x['quality_score'] >= 1)
```

---

### Denormalized Metadata

**Metadata configuration**: All 36 fields (full participant profile)

**Audio configuration**: 10 denormalized fields (convenience for filtering/analysis)

- `participant_id` - Join key to metadata configuration
- `audio_type` - One of 9 recording types
- `audio` - Audio array at 16kHz
- `quality_score` - Manual annotation (0/1/2)
- `record_date` - Recording submission date
- `covid_status` - Health status category
- `age` - Participant age
- `gender` - Participant gender
- `test_status` - COVID test result (p/n/na/ut)
- `vacc` - Vaccination status (y/p/n)
- `country` - Participant country

**Rationale**: Allows quick filtering without joining configurations

```python
# Filter COVID-positive cough recordings without join
cough_covid = ds['audio'].filter(
    lambda x: x['audio_type'] == 'cough-heavy'
              and x['covid_status'].startswith('positive')
)
```

---

### NULL Audio Handling

**Causes**:

1. Audio file missing from `Extracted_data/`
2. File corrupted (librosa load fails)
3. Empty audio file (0 samples)

**Handling**:

- Set audio field to `None`
- Log warning with participant ID and audio type
- Continue processing
- Include in final dataset (allows metadata analysis)

**Expected rate**: ~1-2% of audio files

**Filtering in usage**:

```python
# Exclude NULL audio
ds_clean = ds.filter(lambda x: x['audio'] is not None)
```

---

## Output Structure

### Workspace Directory

```
workspace/
├── metadata_with_quality.csv      # 2,746 rows × 45 columns
├── splits.json                    # {"train": [...], "val": [...], "test": [...]}
├── batches.json                   # {"train": [{batch_id, participants, count}, ...], ...}
├── participant_date_mapping.json  # {participant_id: date_folder, ...}
├── audio_summary.json             # Phase 2a statistics
├── validation_report.json         # Phase 2b results
├── metadata_summary.json          # Phase 3 statistics
├── metadata/
│   ├── train.parquet              # 1,922 rows
│   ├── val.parquet                # 412 rows
│   └── test.parquet               # 412 rows
└── audio/
    ├── train/
    │   ├── audio-train-00001.parquet
    │   ├── audio-train-00002.parquet
    │   ├── ...
    │   └── audio-train-00039.parquet
    ├── val/
    │   ├── audio-val-00001.parquet
    │   ├── ...
    │   └── audio-val-00009.parquet
    └── test/
        ├── audio-test-00001.parquet
        ├── ...
        └── audio-test-00009.parquet
```

### HuggingFace Dataset

```
username/dataset-name
├── metadata/          # Configuration: metadata
│   ├── train.parquet
│   ├── val.parquet
│   └── test.parquet
├── audio/             # Configuration: audio
│   ├── train/
│   │   └── audio-train-*.parquet (auto-detected as single split)
│   ├── val/
│   │   └── audio-val-*.parquet
│   └── test/
│       └── audio-test-*.parquet
└── README.md          # Dataset card with usage examples
```

### Loading the Dataset

```python
from datasets import load_dataset

# Load metadata configuration
metadata = load_dataset("username/dataset-name", "metadata")
# Returns: DatasetDict({train: 1922, val: 412, test: 412})

# Load audio configuration
audio = load_dataset("username/dataset-name", "audio")
# Returns: DatasetDict({train: ~17298, val: ~3708, test: ~3708})

# Access specific split
train_audio = audio["train"]
# Columns: participant_id, audio_type, audio, quality_score, record_date,
#          covid_status, age, gender, test_status, vacc, country
```

---

## Advanced Usage

### Custom Split Ratios

Edit `01_prepare_splits.py`:

```python
# Line 229-231
splits = create_stratified_splits(
    metadata_with_quality,
    test_size_val=0.10,   # 10% validation instead of 15%
    test_size_test=0.20,  # 20% test instead of 15%
    random_state=args.seed
)
```

### Different Audio Types Only

Filter during loading instead of during processing (more flexible):

```python
# Load all audio types
ds = load_dataset("username/dataset-name", "audio")

# Filter for cough only
cough_ds = ds.filter(lambda x: x['audio_type'] in ['cough-heavy', 'cough-shallow'])
```

### Quality-Based Filtering

```python
# Load audio configuration
ds = load_dataset("username/dataset-name", "audio")

# Keep only excellent quality (score 2)
ds_excellent = ds.filter(lambda x: x['quality_score'] == 2)

# Keep good + excellent (scores 1-2)
ds_clean = ds.filter(lambda x: x['quality_score'] >= 1)
```

### Join Audio with Full Metadata

```python
import pandas as pd
from datasets import load_dataset

# Load both configurations
metadata = load_dataset("username/dataset-name", "metadata")
audio = load_dataset("username/dataset-name", "audio")

# Convert to pandas
metadata_df = metadata["train"].to_pandas()
audio_df = audio["train"].to_pandas()

# Join on participant_id
merged = audio_df.merge(
    metadata_df,
    left_on='participant_id',
    right_on='id',
    how='left'
)

# Now you have all 36 metadata fields for each audio sample
print(merged.columns)
```

---

## Dataset Card

The `dataset_card.md` file is uploaded as `README.md` to the HuggingFace dataset repository. It contains:

- Dataset description and statistics
- Two configurations documentation
- Schema tables for all fields
- COVID status categories
- Quality score distribution
- Usage examples (metadata-only, audio classification, multi-task learning, joining)
- Data collection methods
- Known limitations
- Citations (arXiv 2020, Nature 2023)
- License (MIT)
- Ethical considerations

**Location**: `dataset_card.md` → Uploaded as `README.md` in Phase 4
