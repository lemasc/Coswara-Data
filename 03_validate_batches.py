#!/usr/bin/env python3
"""
Phase 2b: Validation
- Verify all batches processed correctly
- Check file integrity and data quality
- Generate validation report
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


def convert_to_json_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def validate_split(
    split_name: str,
    expected_batches: list,
    audio_dir: Path
) -> dict:
    """Validate all batches for a split."""

    print(f"\nValidating {split_name.upper()} split...")

    results = {
        'split': split_name,
        'expected_batches': len(expected_batches),
        'found_batches': 0,
        'expected_participants': sum(b['count'] for b in expected_batches),
        'expected_rows': sum(b['count'] for b in expected_batches) * 9,  # 9 audio types
        'actual_rows': 0,
        'null_audio_count': 0,
        'corrupted_files': [],
        'missing_batches': [],
        'quality_distribution': defaultdict(int),
        'audio_type_distribution': defaultdict(int),
    }

    # Check each expected batch
    for batch_info in tqdm(expected_batches, desc=f"  Checking batches"):
        batch_id = batch_info['batch_id']
        batch_file = audio_dir / split_name / f"audio-{split_name}-{batch_id}.parquet"

        if not batch_file.exists():
            results['missing_batches'].append(batch_id)
            continue

        results['found_batches'] += 1

        # Load and validate batch
        try:
            df = pd.read_parquet(batch_file)

            # Count rows
            results['actual_rows'] += len(df)

            # Count NULL audio
            null_count = df['audio'].isna().sum()
            results['null_audio_count'] += null_count

            # Quality distribution
            for quality in df['quality_score'].dropna():
                results['quality_distribution'][int(quality)] += 1

            # Audio type distribution
            for audio_type in df['audio_type']:
                results['audio_type_distribution'][audio_type] += 1

            # Validate audio column for non-null entries
            for idx, row in df.iterrows():
                if pd.notna(row['audio']):
                    audio = row['audio']
                    if 'sampling_rate' in audio:
                        if audio['sampling_rate'] != 16000:
                            results['corrupted_files'].append(
                                f"{batch_file.name}:{idx} - Invalid sample rate: {audio['sampling_rate']}"
                            )

        except Exception as e:
            results['corrupted_files'].append(f"{batch_file.name} - Error: {str(e)}")

    # Convert defaultdicts to regular dicts
    results['quality_distribution'] = dict(results['quality_distribution'])
    results['audio_type_distribution'] = dict(results['audio_type_distribution'])

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2b: Validate audio batches")
    parser.add_argument('--workspace', type=str, default='./workspace',
                        help='Workspace directory')

    args = parser.parse_args()

    workspace = Path(args.workspace)
    audio_dir = workspace / 'audio'

    # Verify workspace exists
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace}")

    print("="*60)
    print("Phase 2b: Batch Validation")
    print("="*60)

    # Load batch configuration
    print("\nLoading batch configuration...")
    with open(workspace / 'batches.json', 'r') as f:
        batches = json.load(f)

    # Validate each split
    validation_results = {}

    for split_name in ['train', 'val', 'test']:
        expected_batches = batches[split_name]
        results = validate_split(split_name, expected_batches, audio_dir)
        validation_results[split_name] = results

    # Print summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)

    all_pass = True

    for split_name, results in validation_results.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Batches: {results['found_batches']}/{results['expected_batches']}")
        print(f"  Rows: {results['actual_rows']}/{results['expected_rows']}")
        print(f"  NULL audio: {results['null_audio_count']} ({100*results['null_audio_count']/results['actual_rows']:.1f}%)")

        if results['missing_batches']:
            print(f"  ⚠ Missing batches: {results['missing_batches']}")
            all_pass = False

        if results['corrupted_files']:
            print(f"  ⚠ Corrupted files: {len(results['corrupted_files'])}")
            for corrupted in results['corrupted_files'][:5]:
                print(f"    - {corrupted}")
            if len(results['corrupted_files']) > 5:
                print(f"    ... and {len(results['corrupted_files']) - 5} more")
            all_pass = False

        # Quality distribution
        print(f"  Quality distribution:")
        for quality in sorted(results['quality_distribution'].keys()):
            count = results['quality_distribution'][quality]
            print(f"    Quality {quality}: {count}")

        # Row count mismatch
        if results['actual_rows'] != results['expected_rows']:
            print(f"  ⚠ Row count mismatch: expected {results['expected_rows']}, got {results['actual_rows']}")
            all_pass = False

    # Save validation report
    report_file = workspace / 'validation_report.json'
    with open(report_file, 'w') as f:
        # Convert numpy types to JSON-serializable types
        json_serializable_results = convert_to_json_serializable(validation_results)
        json.dump(json_serializable_results, f, indent=2)
    print(f"\n✓ Saved validation report to {report_file}")

    # Overall status
    print("\n" + "="*60)
    if all_pass:
        print("✓ All validations passed!")
    else:
        print("⚠ Some validations failed - please review the report")

    print("="*60)

    if all_pass:
        print(f"\nNext step: Run 04_prepare_metadata.py --workspace {args.workspace}")
    else:
        print("\nPlease fix issues before proceeding to metadata preparation")


if __name__ == '__main__':
    main()
