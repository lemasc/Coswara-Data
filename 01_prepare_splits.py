#!/usr/bin/env python3
"""
Phase 1: Data Preparation
- Load metadata and annotations
- Create stratified train/val/test splits (70/15/15)
- Generate batches for processing
- Build participant-to-date mapping
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# 9 audio types
AUDIO_TYPES = [
    'breathing-deep',
    'breathing-shallow',
    'cough-heavy',
    'cough-shallow',
    'vowel-a',
    'vowel-e',
    'vowel-o',
    'counting-normal',
    'counting-fast'
]


def load_annotations(annotations_dir: Path) -> pd.DataFrame:
    """Load and merge all quality annotation files."""
    print("Loading quality annotations...")

    quality_data = {}

    for audio_type in tqdm(AUDIO_TYPES, desc="Loading annotations"):
        annotation_file = annotations_dir / f"{audio_type}_labels.csv"
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        df = pd.read_csv(annotation_file, skipinitialspace=True)
        df.columns = df.columns.str.strip()

        # Parse FILENAME to extract participant_id
        # Format: {participant_id}_{audio_type}
        df['participant_id'] = df['FILENAME'].str.replace(f'_{audio_type}', '')
        df['quality'] = df['QUALITY']

        # Store quality scores
        for _, row in df.iterrows():
            pid = row['participant_id']
            if pid not in quality_data:
                quality_data[pid] = {}
            quality_data[pid][f'{audio_type.replace("-", "_")}_quality'] = row['quality']

    # Convert to DataFrame
    quality_df = pd.DataFrame.from_dict(quality_data, orient='index')
    quality_df.index.name = 'participant_id'
    quality_df.reset_index(inplace=True)

    print(f"Loaded quality scores for {len(quality_df)} participants")
    return quality_df


def build_participant_date_mapping(extracted_data_dir: Path) -> dict:
    """Build mapping of participant_id -> date folder."""
    print("Building participant-to-date mapping...")

    mapping = {}
    date_folders = sorted([d for d in extracted_data_dir.iterdir() if d.is_dir()])

    for date_folder in tqdm(date_folders, desc="Scanning date folders"):
        date_str = date_folder.name
        participant_folders = [p for p in date_folder.iterdir() if p.is_dir()]

        for participant_folder in participant_folders:
            participant_id = participant_folder.name
            if participant_id in mapping:
                print(f"Warning: Duplicate participant {participant_id} found in {date_str} and {mapping[participant_id]}")
            mapping[participant_id] = date_str

    print(f"Mapped {len(mapping)} participants to date folders")
    return mapping


def create_stratified_splits(df: pd.DataFrame, test_size_val: float, test_size_test: float, random_state: int):
    """Create stratified train/val/test splits."""
    print("Creating stratified splits...")

    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=test_size_test,
        stratify=df['covid_status'],
        random_state=random_state
    )

    # Second split: train vs val
    # Adjust val size relative to train+val
    val_size_adjusted = test_size_val / (1 - test_size_test)
    train, val = train_test_split(
        train_val,
        test_size=val_size_adjusted,
        stratify=train_val['covid_status'],
        random_state=random_state
    )

    print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")

    # Verify stratification
    print("\nCOVID status distribution:")
    for split_name, split_df in [('train', train), ('val', val), ('test', test)]:
        dist = split_df['covid_status'].value_counts(normalize=True)
        print(f"  {split_name}: {dict(dist)}")

    return {
        'train': train['id'].tolist(),
        'val': val['id'].tolist(),
        'test': test['id'].tolist()
    }


def create_batches(splits: dict, batch_size: int, test_mode: int = None):
    """Create batches within each split."""
    print(f"\nCreating batches (size={batch_size})...")

    batches = {}

    for split_name, participant_ids in splits.items():
        split_batches = []

        # Apply test mode limit
        if test_mode:
            max_participants = test_mode * batch_size
            participant_ids = participant_ids[:max_participants]
            print(f"  Test mode: limiting {split_name} to {len(participant_ids)} participants ({test_mode} batches)")

        # Create batches
        for i in range(0, len(participant_ids), batch_size):
            batch_participants = participant_ids[i:i + batch_size]
            batch_id = f"{i // batch_size + 1:05d}"

            split_batches.append({
                'batch_id': batch_id,
                'participants': batch_participants,
                'count': len(batch_participants)
            })

        batches[split_name] = split_batches
        print(f"  {split_name}: {len(split_batches)} batches, {len(participant_ids)} participants")

    return batches


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Prepare splits and batches")
    parser.add_argument('--workspace', type=str, default='./workspace',
                        help='Workspace directory for outputs')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Number of participants per batch')
    parser.add_argument('--test-mode', type=int, default=None,
                        help='Test mode: only process first N batches per split')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting')

    args = parser.parse_args()

    # Paths
    workspace = Path(args.workspace)
    workspace.mkdir(exist_ok=True)

    combined_csv = Path('combined_data.csv')
    annotations_dir = Path('annotations')
    extracted_data_dir = Path('Extracted_data')

    # Verify inputs exist
    if not combined_csv.exists():
        raise FileNotFoundError(f"combined_data.csv not found")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"annotations/ directory not found")
    if not extracted_data_dir.exists():
        raise FileNotFoundError(f"Extracted_data/ directory not found")

    print("="*60)
    print("Phase 1: Data Preparation")
    print("="*60)

    # Step 1: Load metadata
    print("\n[1/5] Loading metadata...")
    metadata_df = pd.read_csv(combined_csv)
    print(f"Loaded {len(metadata_df)} participants from combined_data.csv")

    # Step 2: Load annotations
    print("\n[2/5] Loading annotations...")
    quality_df = load_annotations(annotations_dir)

    # Step 3: Merge metadata with quality scores
    print("\n[3/5] Merging metadata with quality scores...")
    metadata_with_quality = metadata_df.merge(
        quality_df,
        left_on='id',
        right_on='participant_id',
        how='left'
    )

    # Drop duplicate id column
    if 'participant_id' in metadata_with_quality.columns:
        metadata_with_quality.drop(columns=['participant_id'], inplace=True)

    print(f"Merged dataset has {len(metadata_with_quality)} rows and {len(metadata_with_quality.columns)} columns")

    # Step 4: Build participant-to-date mapping
    print("\n[4/5] Building participant-to-date mapping...")
    participant_date_mapping = build_participant_date_mapping(extracted_data_dir)

    # Verify all participants have a date mapping
    missing_mapping = set(metadata_with_quality['id']) - set(participant_date_mapping.keys())
    if missing_mapping:
        print(f"Warning: {len(missing_mapping)} participants not found in Extracted_data/")
        print(f"  First 10: {list(missing_mapping)[:10]}")

    # Step 5: Create splits and batches
    print("\n[5/5] Creating splits and batches...")
    splits = create_stratified_splits(
        metadata_with_quality,
        test_size_val=0.15,
        test_size_test=0.15,
        random_state=args.seed
    )

    batches = create_batches(splits, args.batch_size, args.test_mode)

    # Save outputs
    print("\n" + "="*60)
    print("Saving outputs...")
    print("="*60)

    # Save splits
    splits_file = workspace / 'splits.json'
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"✓ Saved splits to {splits_file}")

    # Save batches
    batches_file = workspace / 'batches.json'
    with open(batches_file, 'w') as f:
        json.dump(batches, f, indent=2)
    print(f"✓ Saved batches to {batches_file}")

    # Save participant-date mapping
    mapping_file = workspace / 'participant_date_mapping.json'
    with open(mapping_file, 'w') as f:
        json.dump(participant_date_mapping, f, indent=2)
    print(f"✓ Saved participant-date mapping to {mapping_file}")

    # Save metadata with quality
    metadata_file = workspace / 'metadata_with_quality.csv'
    metadata_with_quality.to_csv(metadata_file, index=False)
    print(f"✓ Saved metadata with quality to {metadata_file}")

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total participants: {len(metadata_with_quality)}")
    print(f"Participants with date mapping: {len(participant_date_mapping)}")
    print(f"\nSplits:")
    for split_name, participant_ids in splits.items():
        n_batches = len(batches[split_name])
        n_audio_files = len(participant_ids) * 9  # 9 audio files per participant
        print(f"  {split_name}: {len(participant_ids)} participants, {n_batches} batches, {n_audio_files} audio files")

    if args.test_mode:
        print(f"\n⚠ Test mode enabled: only first {args.test_mode} batches per split will be processed")

    print("\n✓ Phase 1 complete!")
    print(f"\nNext step: Run 02_process_audio_batch.py --workspace {args.workspace}")


if __name__ == '__main__':
    main()
