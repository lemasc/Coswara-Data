#!/usr/bin/env python3
"""
Phase 3: Metadata Processing
- Split metadata by train/val/test
- Save as Parquet files
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Prepare metadata splits")
    parser.add_argument('--workspace', type=str, default='./workspace',
                        help='Workspace directory')

    args = parser.parse_args()

    workspace = Path(args.workspace)

    # Verify workspace exists
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace}")

    print("="*60)
    print("Phase 3: Metadata Processing")
    print("="*60)

    start_time = datetime.now()

    # Load metadata
    print("\nLoading metadata...")
    metadata_file = workspace / 'metadata_with_quality.csv'
    metadata_df = pd.read_csv(metadata_file)
    print(f"✓ Loaded {len(metadata_df)} participants")

    # Load splits
    print("\nLoading split assignments...")
    with open(workspace / 'splits.json', 'r') as f:
        splits = json.load(f)

    # Create output directory
    output_dir = workspace / 'metadata'
    output_dir.mkdir(exist_ok=True)

    # Process each split
    print("\nCreating metadata splits...")

    split_stats = {}

    for split_name in ['train', 'val', 'test']:
        participant_ids = splits[split_name]

        # Filter metadata
        split_df = metadata_df[metadata_df['id'].isin(participant_ids)]

        # Verify all participants found
        if len(split_df) != len(participant_ids):
            missing = set(participant_ids) - set(split_df['id'])
            print(f"  ⚠ Warning: {len(missing)} participants not found in metadata for {split_name}")
            if len(missing) <= 10:
                print(f"    Missing IDs: {missing}")

        # Save as Parquet
        output_file = output_dir / f"{split_name}.parquet"
        split_df.to_parquet(output_file, index=False)

        split_stats[split_name] = {
            'expected': len(participant_ids),
            'actual': len(split_df),
            'columns': len(split_df.columns)
        }

        print(f"  ✓ {split_name}: {len(split_df)} rows → {output_file.name}")

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    total_expected = sum(s['expected'] for s in split_stats.values())
    total_actual = sum(s['actual'] for s in split_stats.values())

    print(f"Total participants: {total_actual}/{total_expected}")
    print(f"Columns per file: {split_stats['train']['columns']}")

    for split_name, stats in split_stats.items():
        print(f"  {split_name}: {stats['actual']}/{stats['expected']} participants")

    # Validate
    all_pass = True
    for split_name, stats in split_stats.items():
        if stats['actual'] != stats['expected']:
            print(f"\n⚠ Warning: {split_name} has {stats['actual']} rows but expected {stats['expected']}")
            all_pass = False

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("\n" + "="*60)
    if all_pass:
        print("✓ All metadata splits created successfully!")
    else:
        print("⚠ Some participants missing from metadata")
    print("="*60)
    print(f"Processing time: {duration:.1f}s")

    # Save summary JSON
    summary = {
        'phase': '3_metadata_processing',
        'generated_at': datetime.now().isoformat(),
        'processing_time_seconds': round(duration, 2),
        'configuration': {
            'total_participants': len(metadata_df),
            'total_columns': len(metadata_df.columns)
        },
        'splits': {
            split_name: {
                'expected_participants': stats['expected'],
                'actual_participants': stats['actual'],
                'columns': stats['columns'],
                'missing_participants': stats['expected'] - stats['actual']
            }
            for split_name, stats in split_stats.items()
        },
        'validation': {
            'all_participants_found': all_pass,
            'total_expected': total_expected,
            'total_actual': total_actual
        }
    }

    summary_file = workspace / 'metadata_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to {summary_file.name}")
    print(f"\nNext step: Run 05_upload_to_hf.py --workspace {args.workspace} --repo-id username/dataset-name")


if __name__ == '__main__':
    main()
