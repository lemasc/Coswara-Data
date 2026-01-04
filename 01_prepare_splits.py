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


def verify_participant_files(participant_id: str, expected_date: str, extracted_data_dir: Path) -> dict:
    """
    Verify participant files exist and validate location.

    Returns dict with:
        - status: 'matched_expected' | 'matched_relocated' | 'missing' | 'partial'
        - date_folder: actual date folder (if found)
        - expected_date: date from CSV
        - metadata_date: date from metadata.json (if found)
        - missing_audio: list of missing audio types
        - date_mismatch: bool indicating if CSV date != actual location/metadata
    """
    result = {
        'participant_id': participant_id,
        'expected_date': expected_date,
        'status': 'missing',
        'date_folder': None,
        'metadata_date': None,
        'missing_audio': [],
        'date_mismatch': False
    }

    # Step 1: Check expected location first
    expected_folder = extracted_data_dir / expected_date / participant_id
    participant_folder = None

    if expected_folder.exists() and expected_folder.is_dir():
        participant_folder = expected_folder
        result['date_folder'] = expected_date
    else:
        # Step 2: Search recursively for participant folder
        date_folders = sorted([d for d in extracted_data_dir.iterdir() if d.is_dir()])
        for date_folder in date_folders:
            candidate = date_folder / participant_id
            if candidate.exists() and candidate.is_dir():
                participant_folder = candidate
                result['date_folder'] = date_folder.name
                result['date_mismatch'] = True
                break

    if not participant_folder:
        return result  # Status remains 'missing'

    # Step 3: Validate metadata.json if exists
    metadata_json = participant_folder / 'metadata.json'
    if metadata_json.exists():
        try:
            with open(metadata_json, 'r') as f:
                metadata = json.load(f)
                if 'date' in metadata:
                    result['metadata_date'] = metadata['date']
                    # Extract date portion (YYYY-MM-DD) from ISO format
                    metadata_date_str = metadata['date'][:10].replace('-', '')

                    # If metadata date differs from CSV, flag it
                    if metadata_date_str != expected_date:
                        result['date_mismatch'] = True
                        # Use the date from the folder location (already set in date_folder)
        except Exception as e:
            # Failed to parse metadata.json, continue without it
            pass

    # Step 4: Check all audio files exist
    missing_audio = []
    for audio_type in AUDIO_TYPES:
        audio_file = participant_folder / f"{audio_type}.wav"
        if not audio_file.exists():
            missing_audio.append(audio_type)

    result['missing_audio'] = missing_audio

    # Determine final status
    if len(missing_audio) == len(AUDIO_TYPES):
        result['status'] = 'missing'  # No audio files found
    elif len(missing_audio) > 0:
        result['status'] = 'partial'  # Some audio files missing
    elif result['date_mismatch']:
        result['status'] = 'matched_relocated'  # Found but at wrong location/date
    else:
        result['status'] = 'matched_expected'  # Everything matches

    return result


def build_participant_date_mapping(metadata_df: pd.DataFrame, extracted_data_dir: Path) -> tuple:
    """
    Build mapping of participant_id -> date folder with verification.

    Returns:
        - mapping: dict of participant_id -> date_folder
        - verification_report: detailed verification results
    """
    print("Building participant-to-date mapping with verification...")

    mapping = {}
    verification_results = []

    # Process each participant from metadata
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Verifying participants"):
        participant_id = row['id']
        expected_date = str(row['record_date']).replace('-', '')  # Convert YYYY-MM-DD to YYYYMMDD

        # Verify files
        result = verify_participant_files(participant_id, expected_date, extracted_data_dir)
        verification_results.append(result)

        # Add to mapping if found (excluding fully missing)
        if result['status'] != 'missing':
            mapping[participant_id] = result['date_folder']

    # Generate summary statistics
    status_counts = {}
    for result in verification_results:
        status = result['status']
        status_counts[status] = status_counts.get(status, 0) + 1

    verification_report = {
        'total_participants': len(metadata_df),
        'status_summary': status_counts,
        'mapped_participants': len(mapping),
        'details': verification_results
    }

    print(f"\nVerification Summary:")
    print(f"  Total participants: {verification_report['total_participants']}")
    print(f"  Matched at expected location: {status_counts.get('matched_expected', 0)}")
    print(f"  Matched but relocated: {status_counts.get('matched_relocated', 0)}")
    print(f"  Partially missing audio: {status_counts.get('partial', 0)}")
    print(f"  Completely missing: {status_counts.get('missing', 0)}")
    print(f"  Successfully mapped: {len(mapping)}")

    return mapping, verification_report


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

    # Step 4: Build participant-to-date mapping with verification
    print("\n[4/5] Building participant-to-date mapping with verification...")
    participant_date_mapping, verification_report = build_participant_date_mapping(
        metadata_with_quality,
        extracted_data_dir
    )

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

    # Save verification report
    verification_file = workspace / 'verification_report.json'
    with open(verification_file, 'w') as f:
        json.dump(verification_report, f, indent=2)
    print(f"✓ Saved verification report to {verification_file}")

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

    # Verification details
    status_summary = verification_report['status_summary']
    print(f"\nFile Verification:")
    print(f"  ✓ Matched at expected location: {status_summary.get('matched_expected', 0)}")
    if status_summary.get('matched_relocated', 0) > 0:
        print(f"  ⚠ Matched but relocated: {status_summary.get('matched_relocated', 0)} (date/location mismatch)")
    if status_summary.get('partial', 0) > 0:
        print(f"  ⚠ Partial audio files: {status_summary.get('partial', 0)} (some audio files missing)")
    if status_summary.get('missing', 0) > 0:
        print(f"  ✗ Completely missing: {status_summary.get('missing', 0)}")

    # Show examples of relocated participants
    relocated = [r for r in verification_report['details'] if r['status'] == 'matched_relocated']
    if relocated:
        print(f"\n  Examples of relocated participants (first 5):")
        for r in relocated[:5]:
            print(f"    - {r['participant_id']}: CSV date={r['expected_date']}, found in {r['date_folder']}")

    print(f"\nSplits:")
    for split_name, participant_ids in splits.items():
        n_batches = len(batches[split_name])
        n_audio_files = len(participant_ids) * 9  # 9 audio files per participant
        print(f"  {split_name}: {len(participant_ids)} participants, {n_batches} batches, {n_audio_files} audio files")

    if args.test_mode:
        print(f"\n⚠ Test mode enabled: only first {args.test_mode} batches per split will be processed")

    print("\n✓ Phase 1 complete!")
    print(f"\nVerification report saved to: {verification_file}")
    print(f"Next step: Run 02_process_audio_batch.py --workspace {args.workspace}")


if __name__ == '__main__':
    main()
