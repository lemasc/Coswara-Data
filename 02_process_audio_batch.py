#!/usr/bin/env python3
"""
Phase 2a: Audio Processing
- Load audio files for each batch
- Verify integrity and downsample to 16kHz
- Create Parquet files with Audio features
- Support parallel processing
"""

import argparse
import json
import os
import warnings
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any
from datetime import datetime

import librosa
import numpy as np
import pandas as pd
from datasets import Dataset, Audio, Features, Value
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

# Target sampling rate
TARGET_SR = 16000


def load_and_resample_audio(audio_path: Path, target_sr: int = TARGET_SR) -> tuple:
    """
    Load audio file and resample to target sampling rate.

    Returns:
        (audio_array, sampling_rate, error_message)
        If successful: (array, 16000, None)
        If failed: (None, None, error_message)
    """
    try:
        # Load audio at native sample rate
        # Note: Dataset has mixed sample rates (48kHz, 44.1kHz, 16kHz)
        audio, sr = librosa.load(audio_path, sr=None)

        # Verify audio has content
        if len(audio) == 0:
            return None, None, "Empty audio file"

        # Resample to target 16kHz if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        return audio, target_sr, None

    except Exception as e:
        return None, None, str(e)


def process_participant(args: tuple) -> List[Dict[str, Any]]:
    """
    Process all audio files for a single participant.

    Args:
        args: (participant_id, date_folder, metadata_row, extracted_data_dir)

    Returns:
        List of rows (one per audio type)
    """
    participant_id, date_folder, metadata_row, extracted_data_dir = args

    rows = []
    participant_dir = Path(extracted_data_dir) / date_folder / participant_id

    for audio_type in AUDIO_TYPES:
        # Construct audio file path
        audio_filename = f"{audio_type}.wav"
        audio_path = participant_dir / audio_filename

        # Load and resample audio
        if audio_path.exists():
            audio_array, sr, error = load_and_resample_audio(audio_path)

            if error:
                warnings.warn(f"Error processing {participant_id}/{audio_type}: {error}")
                audio_dict = None
            else:
                audio_dict = {
                    'array': audio_array,  # Keep as numpy array for HuggingFace Audio feature
                    'sampling_rate': sr
                }
        else:
            warnings.warn(f"Audio file not found: {audio_path}")
            audio_dict = None

        # Get quality score
        quality_col = f"{audio_type.replace('-', '_')}_quality"
        quality_score = metadata_row.get(quality_col, None)

        # Create row with denormalized metadata
        row = {
            # Identifiers
            'participant_id': participant_id,
            'audio_type': audio_type,

            # Audio
            'audio': audio_dict,

            # Quality
            'quality_score': int(quality_score) if pd.notna(quality_score) else None,

            # Denormalized key metadata fields (10 fields total above)
            'record_date': metadata_row.get('record_date', None),
            'covid_status': metadata_row.get('covid_status', None),
            'age': int(metadata_row['a']) if pd.notna(metadata_row.get('a')) else None,
            'gender': metadata_row.get('g', None),
            'test_status': metadata_row.get('test_status', None),
            'vacc': metadata_row.get('vacc', None),
            'country': metadata_row.get('l_c', None),
        }

        rows.append(row)

    return rows


def process_batch(
    batch_info: dict,
    split_name: str,
    metadata_df: pd.DataFrame,
    participant_date_mapping: dict,
    extracted_data_dir: Path,
    output_dir: Path,
    num_workers: int = 1
):
    """Process a single batch and save as Parquet."""

    batch_id = batch_info['batch_id']
    participant_ids = batch_info['participants']

    print(f"\nProcessing {split_name}/batch-{batch_id} ({len(participant_ids)} participants)...")

    # Prepare arguments for parallel processing
    process_args = []
    for participant_id in participant_ids:
        # Get metadata row
        metadata_row = metadata_df[metadata_df['id'] == participant_id]
        if metadata_row.empty:
            warnings.warn(f"Metadata not found for participant {participant_id}")
            continue

        metadata_row = metadata_row.iloc[0].to_dict()

        # Get date folder
        date_folder = participant_date_mapping.get(participant_id)
        if not date_folder:
            warnings.warn(f"Date folder not found for participant {participant_id}")
            continue

        process_args.append((participant_id, date_folder, metadata_row, extracted_data_dir))

    # Process participants in parallel
    all_rows = []

    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_participant, process_args),
                total=len(process_args),
                desc=f"  Processing participants",
                leave=False
            ))
            for participant_rows in results:
                all_rows.extend(participant_rows)
    else:
        # Single-threaded processing
        for args in tqdm(process_args, desc=f"  Processing participants", leave=False):
            participant_rows = process_participant(args)
            all_rows.extend(participant_rows)

    # Convert to Dataset
    print(f"  Creating dataset from {len(all_rows)} rows...")

    # Organize data by columns
    data_dict = {
        'participant_id': [],
        'audio_type': [],
        'audio': [],
        'quality_score': [],
        'record_date': [],
        'covid_status': [],
        'age': [],
        'gender': [],
        'test_status': [],
        'vacc': [],
        'country': [],
    }

    for row in all_rows:
        for key in data_dict.keys():
            data_dict[key].append(row[key])

    # Define explicit schema to avoid type inference issues
    features = Features({
        'participant_id': Value('string'),
        'audio_type': Value('string'),
        'audio': Audio(sampling_rate=TARGET_SR),
        'quality_score': Value('int64'),
        'record_date': Value('string'),
        'covid_status': Value('string'),
        'age': Value('int64'),
        'gender': Value('string'),
        'test_status': Value('string'),
        'vacc': Value('string'),
        'country': Value('string'),
    })

    # Create Dataset with explicit schema
    dataset = Dataset.from_dict(data_dict, features=features)

    # Save as Parquet
    output_file = output_dir / f"audio-{split_name}-{batch_id}.parquet"
    dataset.to_parquet(output_file)

    print(f"  ✓ Saved {len(dataset)} rows to {output_file.name}")

    # Count null audio
    null_count = sum(1 for x in data_dict['audio'] if x is None)
    if null_count > 0:
        print(f"  ⚠ Warning: {null_count} audio files were NULL due to errors")

    return len(dataset), null_count


def main():
    parser = argparse.ArgumentParser(description="Phase 2a: Process audio batches")
    parser.add_argument('--workspace', type=str, default='./workspace',
                        help='Workspace directory')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'val', 'test', 'all'],
                        help='Which split to process')
    parser.add_argument('--batch', type=str, default=None,
                        help='Specific batch ID to process (e.g., "00001")')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count - 1)')
    parser.add_argument('--extracted-data', type=str, default='./Extracted_data',
                        help='Path to Extracted_data directory')

    args = parser.parse_args()

    # Set default workers
    if args.num_workers is None:
        args.num_workers = max(1, cpu_count() - 1)

    # Paths
    workspace = Path(args.workspace)
    extracted_data_dir = Path(args.extracted_data)

    # Verify inputs
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace}")
    if not extracted_data_dir.exists():
        raise FileNotFoundError(f"Extracted_data not found: {extracted_data_dir}")

    # Load configuration files
    print("="*60)
    print("Phase 2a: Audio Processing")
    print("="*60)

    print("\nLoading configuration files...")
    with open(workspace / 'batches.json', 'r') as f:
        batches = json.load(f)

    with open(workspace / 'participant_date_mapping.json', 'r') as f:
        participant_date_mapping = json.load(f)

    metadata_df = pd.read_csv(workspace / 'metadata_with_quality.csv')

    print(f"✓ Loaded configuration")
    print(f"  Metadata: {len(metadata_df)} participants")
    print(f"  Date mappings: {len(participant_date_mapping)} participants")
    print(f"  Workers: {args.num_workers}")

    # Determine which splits to process
    if args.split == 'all':
        splits_to_process = ['train', 'val', 'test']
    else:
        splits_to_process = [args.split]

    # Process batches
    total_rows = 0
    total_null = 0
    start_time = datetime.now()

    # Track statistics for summary
    split_stats = {}

    for split_name in splits_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} split")
        print(f"{'='*60}")

        split_batches = batches[split_name]

        # Create output directory
        output_dir = workspace / 'audio' / split_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filter to specific batch if requested
        if args.batch:
            split_batches = [b for b in split_batches if b['batch_id'] == args.batch]
            if not split_batches:
                print(f"Warning: Batch {args.batch} not found in {split_name} split")
                continue

        # Track split-level stats
        split_rows = 0
        split_null = 0
        batch_details = []

        # Process each batch
        for batch_info in split_batches:
            rows, null = process_batch(
                batch_info,
                split_name,
                metadata_df,
                participant_date_mapping,
                extracted_data_dir,
                output_dir,
                args.num_workers
            )
            total_rows += rows
            total_null += null
            split_rows += rows
            split_null += null

            batch_details.append({
                'batch_id': batch_info['batch_id'],
                'participants': len(batch_info['participants']),
                'rows': rows,
                'null_audio': null
            })

        # Store split statistics
        split_stats[split_name] = {
            'total_rows': split_rows,
            'null_audio': split_null,
            'null_percentage': round(100 * split_null / split_rows, 2) if split_rows > 0 else 0,
            'batches': len(split_batches),
            'batch_details': batch_details
        }

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Total rows processed: {total_rows}")
    print(f"Total NULL audio: {total_null} ({100*total_null/total_rows:.1f}%)")
    print(f"Processing time: {duration:.1f}s")

    # Save summary JSON
    summary = {
        'phase': '2a_audio_processing',
        'generated_at': datetime.now().isoformat(),
        'processing_time_seconds': round(duration, 2),
        'configuration': {
            'num_workers': args.num_workers,
            'splits_processed': splits_to_process,
            'batch_filter': args.batch
        },
        'totals': {
            'total_rows': total_rows,
            'total_null_audio': total_null,
            'null_percentage': round(100 * total_null / total_rows, 2) if total_rows > 0 else 0
        },
        'splits': split_stats
    }

    summary_file = workspace / 'audio_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to {summary_file.name}")
    print(f"\n✓ Phase 2a complete!")
    print(f"\nNext step: Run 03_validate_batches.py --workspace {args.workspace}")


if __name__ == '__main__':
    main()
