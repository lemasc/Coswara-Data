---
license: mit
task_categories:
  - audio-classification
  - medical
tags:
  - covid-19
  - health
  - audio
  - medical-diagnosis
  - speech
  - breathing
  - cough
size_categories:
  - 10K<n<100K
language:
  - en
pretty_name: Coswara COVID-19 Audio Dataset
---

# Coswara COVID-19 Audio Dataset

## Dataset Description

The Coswara dataset is a crowdsourced collection of respiratory sounds and speech recordings for COVID-19 diagnosis research, collected by the Indian Institute of Science (IISc) Bangalore. The dataset contains audio recordings of breathing, cough, and speech from 2,746 participants collected between April 2020 and February 2022.

### Dataset Summary

- **Total Participants:** 2,746
- **Total Audio Files:** 24,718 (9 recordings per participant)
- **Collection Period:** April 2020 - February 2022
- **Audio Format:** 16kHz WAV (resampled from mixed source rates: 48kHz, 44.1kHz, 16kHz)
- **Geographic Coverage:** 20+ countries (91.6% India)
- **COVID-19 Positive:** 681 participants (24.8%)

### Supported Tasks

- **Audio Classification:** COVID-19 detection from respiratory sounds
- **Multi-task Learning:** Joint prediction using multiple audio types
- **Health Status Classification:** 8-class health status prediction
- **Symptom Analysis:** Correlation between audio features and COVID symptoms

## Dataset Structure

### Configurations

The dataset provides two configurations (subsets):

#### 1. `metadata` Configuration

Participant health and demographic information.

**Splits:**

- `train`: 1,922 participants
- `val`: 412 participants
- `test`: 412 participants

**Features:** 36 fields including COVID status, demographics, symptoms, and comorbidities

#### 2. `audio` Configuration

Individual audio recordings with quality annotations and denormalized metadata.

**Splits:**

- `train`: ~17,298 audio files (1,922 participants × 9)
- `val`: ~3,708 audio files (412 participants × 9)
- `test`: ~3,708 audio files (412 participants × 9)

**Features:** Audio arrays (16kHz) + quality scores + 10 key metadata fields

### Audio Types

Each participant provided 9 recordings:

1. **breathing-deep** - Deep breathing sounds
2. **breathing-shallow** - Shallow breathing sounds
3. **cough-heavy** - Heavy cough sounds
4. **cough-shallow** - Shallow cough sounds
5. **vowel-a** - Vowel 'a' pronunciation
6. **vowel-e** - Vowel 'e' pronunciation
7. **vowel-o** - Vowel 'o' pronunciation
8. **counting-normal** - Counting at normal speed
9. **counting-fast** - Counting at fast speed

## Schema

### `metadata` Configuration

| Field                       | Type   | Description                                                                  |
| --------------------------- | ------ | ---------------------------------------------------------------------------- |
| `id`                        | string | Unique participant identifier (hash)                                         |
| `a`                         | int64  | Age in years                                                                 |
| `g`                         | string | Gender (male/female/other)                                                   |
| `covid_status`              | string | Health status (8 categories, see below)                                      |
| `record_date`               | date   | Recording submission date                                                    |
| `test_status`               | string | COVID test result (p=positive, n=negative, na=not applicable, ut=under test) |
| `test_date`                 | date   | Date of COVID test                                                           |
| `testType`                  | string | Test type (RAT/RT-PCR)                                                       |
| `vacc`                      | string | Vaccination status (y=fully, p=partially, n=not vaccinated)                  |
| `l_c`                       | string | Country                                                                      |
| `l_s`                       | string | State/province                                                               |
| `l_l`                       | string | Locality/city                                                                |
| `ep`                        | string | English proficiency (y/n)                                                    |
| `rU`                        | string | Returning user (y/n)                                                         |
| **Symptoms**                |        |                                                                              |
| `fever`                     | string | Fever symptom                                                                |
| `cough`                     | string | Cough symptom                                                                |
| `bd`                        | string | Breathing difficulties                                                       |
| `loss_of_smell`             | string | Loss of smell symptom                                                        |
| `st`                        | string | Sore throat                                                                  |
| `ftg`                       | string | Fatigue                                                                      |
| `mp`                        | string | Muscle pain                                                                  |
| `diarrhoea`                 | string | Diarrhea symptom                                                             |
| **Comorbidities**           |        |                                                                              |
| `smoker`                    | string | Smoking status                                                               |
| `cold`                      | string | Cold                                                                         |
| `ht`                        | string | Hypertension                                                                 |
| `diabetes`                  | string | Diabetes                                                                     |
| `asthma`                    | string | Asthma                                                                       |
| `ihd`                       | string | Ischemic heart disease                                                       |
| `cld`                       | string | Chronic lung disease                                                         |
| `pneumonia`                 | string | Pneumonia                                                                    |
| `others_resp`               | string | Other respiratory conditions                                                 |
| `others_preexist`           | string | Other pre-existing conditions                                                |
| **CT Scan**                 |        |                                                                              |
| `ctScan`                    | string | CT scan performed                                                            |
| `ctDate`                    | date   | CT scan date                                                                 |
| `ctScore`                   | string | CT severity score                                                            |
| **Quality Scores**          |        |                                                                              |
| `breathing_deep_quality`    | int64  | Quality label (0=bad, 1=good, 2=excellent)                                   |
| `breathing_shallow_quality` | int64  | Quality label                                                                |
| `cough_heavy_quality`       | int64  | Quality label                                                                |
| `cough_shallow_quality`     | int64  | Quality label                                                                |
| `vowel_a_quality`           | int64  | Quality label                                                                |
| `vowel_e_quality`           | int64  | Quality label                                                                |
| `vowel_o_quality`           | int64  | Quality label                                                                |
| `counting_normal_quality`   | int64  | Quality label                                                                |
| `counting_fast_quality`     | int64  | Quality label                                                                |

### `audio` Configuration

| Field            | Type   | Description                                            |
| ---------------- | ------ | ------------------------------------------------------ |
| `participant_id` | string | Unique participant identifier (joins with metadata)    |
| `audio_type`     | string | Type of recording (breathing-deep, cough-heavy, etc.)  |
| `audio`          | Audio  | Audio array at 16kHz sampling rate                     |
| `quality_score`  | int64  | Manual quality annotation (0=bad, 1=good, 2=excellent) |
| `record_date`    | date   | Recording submission date                              |
| `covid_status`   | string | Participant's COVID health status                      |
| `age`            | int64  | Participant's age                                      |
| `gender`         | string | Participant's gender                                   |
| `test_status`    | string | COVID test result                                      |
| `vacc`           | string | Vaccination status                                     |
| `country`        | string | Participant's country                                  |

### COVID Status Categories

| Status                        | Description                              | Count | Percentage |
| ----------------------------- | ---------------------------------------- | ----- | ---------- |
| `healthy`                     | Healthy individuals                      | 1,433 | 52.2%      |
| `positive_mild`               | COVID-19 positive with mild symptoms     | 426   | 15.5%      |
| `no_resp_illness_exposed`     | Exposed but no respiratory illness       | 248   | 9.0%       |
| `positive_moderate`           | COVID-19 positive with moderate symptoms | 165   | 6.0%       |
| `resp_illness_not_identified` | Respiratory illness, cause unknown       | 157   | 5.7%       |
| `recovered_full`              | Fully recovered from COVID-19            | 146   | 5.3%       |
| `positive_asymp`              | COVID-19 positive, asymptomatic          | 90    | 3.3%       |
| `under_validation`            | Status under validation                  | 81    | 3.0%       |

### Quality Score Distribution

Quality labels were manually annotated for each audio file:

- **0 (Bad):** Poor quality, significant noise/distortion (excluded from classification)
- **1 (Good):** Acceptable quality for analysis
- **2 (Excellent):** High quality, clear audio

**Example distribution (breathing-deep):**

- Quality 2: 71.5%
- Quality 1: 12.1%
- Quality 0: 16.5%

## Usage Examples

### Load Metadata Only

```python
from datasets import load_dataset

# Load metadata configuration
metadata = load_dataset("username/dataset-name", "metadata")

# Access splits
train_metadata = metadata["train"]
val_metadata = metadata["val"]
test_metadata = metadata["test"]

# Convert to pandas for analysis
import pandas as pd
df = train_metadata.to_pandas()

# Analyze COVID status distribution
print(df['covid_status'].value_counts())
```

### Load Audio for COVID Classification

```python
from datasets import load_dataset

# Load audio configuration
audio_ds = load_dataset("username/dataset-name", "audio")

# Filter for high-quality cough recordings
cough_ds = audio_ds["train"].filter(
    lambda x: x['audio_type'] in ['cough-heavy', 'cough-shallow']
              and x['quality_score'] >= 1
)

# Separate by COVID status
covid_positive = cough_ds.filter(
    lambda x: x['covid_status'].startswith('positive')
)
healthy = cough_ds.filter(
    lambda x: x['covid_status'] == 'healthy'
)

print(f"COVID-19 positive samples: {len(covid_positive)}")
print(f"Healthy samples: {len(healthy)}")
```

### Train Audio Classifier with Wav2Vec2

```python
from datasets import load_dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# Load breathing recordings
ds = load_dataset("username/dataset-name", "audio")
breathing_ds = ds["train"].filter(
    lambda x: x['audio_type'] == 'breathing-deep' and x['quality_score'] >= 1
)

# Create binary labels (COVID positive vs healthy)
def create_labels(example):
    example['label'] = 1 if example['covid_status'].startswith('positive') else 0
    return example

breathing_ds = breathing_ds.map(create_labels)

# Load pretrained model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=2
)

# Preprocess audio
def preprocess(batch):
    audio = [x["array"] for x in batch["audio"]]
    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )
    return inputs

# ... continue with training loop
```

### Multi-Task Learning (All Audio Types)

```python
from datasets import load_dataset

# Load all audio types
audio_ds = load_dataset("username/dataset-name", "audio")

# Filter for high quality
train_ds = audio_ds["train"].filter(lambda x: x['quality_score'] >= 1)

# Group by participant for multi-task learning
from collections import defaultdict

participant_data = defaultdict(list)
for example in train_ds:
    participant_data[example['participant_id']].append(example)

# Each participant now has up to 9 audio samples
print(f"Participants with complete recordings: {sum(1 for v in participant_data.values() if len(v) == 9)}")
```

### Join Audio with Full Metadata

```python
from datasets import load_dataset
import pandas as pd

# Load both configurations
metadata = load_dataset("username/dataset-name", "metadata")
audio = load_dataset("username/dataset-name", "audio")

# Convert to pandas
metadata_df = metadata["train"].to_pandas()
audio_df = audio["train"].to_pandas()

# Join for full metadata access
merged = audio_df.merge(
    metadata_df,
    left_on='participant_id',
    right_on='id',
    how='left',
    suffixes=('', '_full')
)

# Now you have all 36 metadata fields for each audio sample
print(f"Merged dataset shape: {merged.shape}")
```

## Dataset Splits

Splits are stratified by `covid_status` to maintain class distribution:

- **Train (70%):** 1,922 participants, ~17,298 audio files
- **Val (15%):** 412 participants, ~3,708 audio files
- **Test (15%):** 412 participants, ~3,708 audio files

**Important:** All 9 audio recordings from the same participant are kept in the same split to prevent data leakage.

## Data Collection

### Collection Method

- **Platform:** Web-based and mobile app submission
- **Period:** April 13, 2020 - February 24, 2022
- **Participants:** Crowdsourced volunteers worldwide
- **Recording Environment:** Uncontrolled (home/personal devices)
- **Consent:** All participants provided informed consent

### Quality Control

- Manual quality annotation by trained annotators
- 3-point scale: 0 (bad), 1 (good), 2 (excellent)
- Annotations available for filtering low-quality samples

### Known Limitations

1. **Uncontrolled Recording Environment:** Varying background noise, device quality
2. **Mixed Sample Rates:** Original recordings have variable sample rates (48kHz, 44.1kHz, 16kHz) due to different recording devices; all resampled to 16kHz for consistency
3. **Class Imbalance:** 52% healthy vs 25% COVID-positive
4. **Geographic Bias:** 91.6% from India
5. **Self-Reported Data:** Some metadata fields rely on participant reporting
6. **Temporal Coverage:** Primarily pre-vaccination era (2020) and Delta variant period (2021)
7. **Missing Data:** Some audio files may be NULL due to collection errors (~1-2%)

## Citation

If you use this dataset, please cite:

### Original Paper

```bibtex
@article{sharma2020coswara,
  title={Coswara--A Database of Breathing, Cough, and Voice Sounds for COVID-19 Diagnosis},
  author={Sharma, Neeraj and Krishnan, Prashant and Kumar, Rohit and Ramoji, Shreyas and Chetupalli, Srikanth Raj and Ghosh, Pravin and others},
  journal={arXiv preprint arXiv:2005.10548},
  year={2020}
}
```

### Nature Scientific Data Publication

```bibtex
@article{sharma2023coswara,
  title={Coswara: a respiratory sounds and symptoms dataset for remote screening of SARS-CoV-2 infection},
  author={Sharma, Neeraj Kumar and Chetupalli, Srikanth Raj and Bhattacharyya, Debarpan and others},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={397},
  year={2023},
  publisher={Nature Publishing Group}
}
```

### Links

- **Paper (arXiv):** https://arxiv.org/abs/2005.10548
- **Paper (Nature):** https://www.nature.com/articles/s41597-023-02266-0
- **Project Website:** https://coswara.iisc.ac.in/

## License

This dataset is released under the **MIT License**.

## Ethical Considerations

- All participants provided informed consent for data collection and research use
- Participant identifiers are anonymized hash strings
- Geographic data is limited to country/state/city level
- Researchers should consider geographic and demographic biases when generalizing findings
- This dataset is for research purposes only and should not be used for clinical diagnosis without proper validation

## Acknowledgments

This dataset was collected and curated by the Indian Institute of Science (IISc) Bangalore. We thank all the volunteers who contributed their recordings to support COVID-19 research.
