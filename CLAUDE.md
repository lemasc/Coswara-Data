# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Coswara-Data is a COVID-19 diagnostic research dataset from IISc Bangalore containing audio recordings (breathing, cough, speech) and metadata from crowdsourced participants. Published in Nature Scientific Data (2023). The repository contains date-wise compressed archives of participant data and technical validation code for classification tasks.

## Data Structure

### Repository Layout

- **Date folders** (`YYYYMMDD/`): Compressed archives (`.tar.gz.aa`, `.tar.gz.ab`, etc.) containing participant audio recordings and metadata CSV files
- **`combined_data.csv`**: Aggregated metadata for all participants across all dates
- **`csv_labels_legend.json`**: Field definitions for metadata columns (covid_status, health conditions, demographics)
- **`annotations/`**: Audio quality labels (0=bad, 1=good, 2=excellent) for each recording type
- **`Extracted_data/`**: Output directory for extracted audio files (created by `extract_data.py`)
- **`technical_validation/`**: Classification experiments (9-class and gender classification)

### Audio Recording Types

Nine sound categories per participant:
- `breathing-deep`, `breathing-shallow`
- `cough-heavy`, `cough-shallow`
- `vowel-a`, `vowel-e`, `vowel-o`
- `counting-normal`, `counting-fast`

### Metadata Fields

Key fields in CSV files (see `csv_labels_legend.json` for complete list):
- **Health**: `covid_status`, `test_status`, `test_date`, `vacc` (vaccination)
- **Demographics**: `a` (age), `g` (gender), `l_c` (country), `l_s` (state)
- **Symptoms**: `fever`, `cough`, `bd` (breathing difficulties), `loss_of_smell`, `st` (sore throat), `ftg` (fatigue)
- **Comorbidities**: `diabetes`, `asthma`, `ht` (hypertension), `smoker`, `ihd`, `cld`, `pneumonia`

## Common Commands

### Data Extraction

```bash
# Extract all compressed date folders to Extracted_data/
python extract_data.py
```

The script:
- Creates `Extracted_data/` if it doesn't exist
- Concatenates multi-part tar.gz files (`cat *.tar.gz.* | tar -xvz`)
- Extracts only folders not already in `Extracted_data/`

### Technical Validation Workflows

Navigate to `technical_validation/9_class_classification/` or `technical_validation/gender_classification/` and run:

#### Feature Extraction
```bash
# Extract features for a specific audio category
./run_feature_extraction.sh <audio_category>

# Example for deep breathing recordings:
./run_feature_extraction.sh breathing-deep
```

This pipeline:
1. Prepares file lists from annotation labels (`local/prepare_list.py`)
2. Extracts acoustic features using librosa/torchaudio (`local/feature_extraction.py`)
3. Filters files with failed feature extraction (`local/filter_list.py`)

Features are saved as pickle files in `feats/<audio_category>/`.

#### Classification
```bash
# Run classification experiment
./run_classification.sh
```

Uses all nine audio categories by default (see `audiocategory` variable in script). Runs RandomForest classifier with train/test splits and outputs results to `results/`.

### Configuration Files

- **`conf/feature.conf`**: Feature extraction settings
  - `feature_type`: `mfcc` or `logMelSpec`
  - `sampling_rate`: 44100 Hz
  - Window parameters, SAD (sound activity detection) thresholds
  - MFCC/mel-spectrogram parameters (n_mfcc, n_mels, deltas)

- **`conf/classification.conf`**: Classifier settings
  - `classifier`: RandomForest (default)
  - Model hyperparameters (n_estimators, criterion)

## Technical Validation Architecture

### Pipeline Flow

1. **Annotation-based filtering**: Quality labels in `annotations/LABELS/<category>_labels.csv` filter recordings (quality 1-2 are kept, 0 discarded)
2. **Path resolution**: `path_files/wav.scp` maps recording IDs to file paths in `Extracted_data/`
3. **Feature extraction** (`local/feature_extraction.py`):
   - Reads audio with librosa at 44.1kHz
   - Applies waveform normalization and SAD to remove silence
   - Computes MFCC or log-mel spectrogram features with deltas
   - Saves features as pickled numpy arrays
4. **Classification** (`local/classification.py`):
   - Loads features and labels
   - Trains sklearn RandomForest with PCA dimensionality reduction
   - Evaluates on train/test splits with fixed random seed (42)

### Directory Structure in Technical Validation

```
technical_validation/<task>/
├── annotations/LABELS/          # Quality annotations per audio type
├── conf/                        # feature.conf, classification.conf
├── data/<category>/             # Generated: file lists, labels, bad_ids
├── feats/<category>/            # Generated: pickled feature matrices
├── local/                       # Python scripts
│   ├── prepare_list.py          # Filter by quality annotations
│   ├── feature_extraction.py    # Extract acoustic features
│   ├── filter_list.py           # Remove failed extractions
│   └── classification.py        # Train and evaluate model
├── path_files/wav.scp           # Recording ID to file path mapping
├── results/                     # Classification outputs
└── run_*.sh                     # Pipeline scripts
```

## Dependencies

This repository uses Python with audio processing libraries. Based on the code:

**Core libraries**:
- `numpy`, `pandas`
- `librosa` (audio loading, feature extraction)
- `torchaudio`, `torch` (PyTorch for audio processing)
- `scikit-learn` (PCA, StandardScaler, RandomForestClassifier)
- `pickle` (feature serialization)
- `matplotlib` (visualization in notebooks)

**Note**: No `requirements.txt` is provided. Install dependencies as needed based on import errors.

## Important Notes

- **Data extraction is destructive**: `extract_data.py` uses shell commands (`cat`, `tar`) and will skip already-extracted folders
- **Random seeds**: Classification scripts use `SEED=42` for reproducibility
- **Audio format**: Expects WAV files at 44.1kHz (resampled during feature extraction if needed)
- **Quality filtering**: Only recordings with quality labels 1 or 2 are used for classification
- **Sound Activity Detection (SAD)**: Automatically removes silence/noise regions based on energy threshold before feature extraction
- **Feature caching**: Features are saved as pickle files to avoid recomputation; delete `feats/` directory to regenerate
- **Multi-part archives**: Date folders contain split tar.gz files (`.aa`, `.ab`, etc.) that must be concatenated before extraction

## Citation

When using this dataset, cite the paper:
Coswara - A Database of Breathing, Cough, and Voice Sounds for COVID-19 Diagnosis
https://arxiv.org/abs/2005.10548

Dataset published in Nature Scientific Data (2023):
https://www.nature.com/articles/s41597-023-02266-0
