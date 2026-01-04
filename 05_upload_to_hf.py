#!/usr/bin/env python3
"""
Phase 4: Upload to Hugging Face
- Upload metadata and audio configurations
- Upload dataset card
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, login, whoami
from tqdm import tqdm


def verify_authentication(token: str = None):
    """Verify HuggingFace authentication."""
    try:
        if token:
            login(token=token)

        user_info = whoami()
        print(f"✓ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        print("\nPlease authenticate using one of these methods:")
        print("  1. Run: huggingface-cli login")
        print("  2. Pass token via --token argument")
        print("  3. Set HF_TOKEN environment variable")
        return False


def verify_workspace(workspace: Path) -> dict:
    """Verify workspace has all required files."""
    print("\nVerifying workspace...")

    checks = {
        'metadata': {
            'train': workspace / 'metadata' / 'train.parquet',
            'val': workspace / 'metadata' / 'val.parquet',
            'test': workspace / 'metadata' / 'test.parquet',
        },
        'audio': {
            'train': workspace / 'audio' / 'train',
            'val': workspace / 'audio' / 'val',
            'test': workspace / 'audio' / 'test',
        }
    }

    results = {
        'metadata_files': [],
        'audio_folders': [],
        'total_audio_files': 0,
        'all_pass': True
    }

    # Check metadata files
    for split_name, file_path in checks['metadata'].items():
        if file_path.exists():
            results['metadata_files'].append(str(file_path.relative_to(workspace)))
            print(f"  ✓ metadata/{split_name}.parquet")
        else:
            print(f"  ✗ Missing: metadata/{split_name}.parquet")
            results['all_pass'] = False

    # Check audio folders and count files
    for split_name, folder_path in checks['audio'].items():
        if folder_path.exists():
            parquet_files = list(folder_path.glob('*.parquet'))
            results['audio_folders'].append(split_name)
            results['total_audio_files'] += len(parquet_files)
            print(f"  ✓ audio/{split_name}/ ({len(parquet_files)} parquet files)")
        else:
            print(f"  ✗ Missing: audio/{split_name}/")
            results['all_pass'] = False

    return results


def upload_files(
    workspace: Path,
    repo_id: str,
    api: HfApi,
    dry_run: bool = False
):
    """Upload all files to HuggingFace."""

    if dry_run:
        print("\n" + "="*60)
        print("DRY RUN MODE - No files will be uploaded")
        print("="*60)

    # Upload metadata config
    print("\n[1/3] Uploading metadata configuration...")
    metadata_dir = workspace / 'metadata'

    if dry_run:
        print(f"  Would upload: {metadata_dir} → metadata/")
    else:
        api.upload_folder(
            folder_path=str(metadata_dir),
            path_in_repo="metadata",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"  ✓ Uploaded metadata configuration")

    # Upload audio config
    print("\n[2/3] Uploading audio configuration...")
    audio_dir = workspace / 'audio'

    if dry_run:
        for split in ['train', 'val', 'test']:
            split_dir = audio_dir / split
            if split_dir.exists():
                n_files = len(list(split_dir.glob('*.parquet')))
                print(f"  Would upload: {split_dir} → audio/{split}/ ({n_files} files)")
    else:
        api.upload_folder(
            folder_path=str(audio_dir),
            path_in_repo="audio",
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"  ✓ Uploaded audio configuration")

    # Upload dataset card
    print("\n[3/3] Uploading dataset card...")
    dataset_card = Path('dataset_card.md')

    if not dataset_card.exists():
        print(f"  ⚠ Warning: dataset_card.md not found, skipping")
    else:
        if dry_run:
            print(f"  Would upload: dataset_card.md → README.md")
        else:
            api.upload_file(
                path_or_fileobj=str(dataset_card),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset"
            )
            print(f"  ✓ Uploaded dataset card")


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Upload to Hugging Face")
    parser.add_argument('--workspace', type=str, default='./workspace',
                        help='Workspace directory')
    parser.add_argument('--repo-id', type=str, required=True,
                        help='HuggingFace repository ID (username/dataset-name)')
    parser.add_argument('--token', type=str, default=None,
                        help='HuggingFace API token (or use huggingface-cli login)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate without uploading')
    parser.add_argument('--create-repo', action='store_true',
                        help='Create repository if it does not exist')

    args = parser.parse_args()

    workspace = Path(args.workspace)

    print("="*60)
    print("Phase 4: Upload to Hugging Face")
    print("="*60)

    # Verify workspace
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace}")

    verify_results = verify_workspace(workspace)

    if not verify_results['all_pass']:
        print("\n✗ Workspace verification failed!")
        print("Please ensure all previous phases completed successfully")
        return

    print(f"\n✓ Workspace verified")
    print(f"  Metadata files: {len(verify_results['metadata_files'])}")
    print(f"  Audio parquet files: {verify_results['total_audio_files']}")

    # Verify authentication
    print("\n" + "="*60)
    print("Authentication")
    print("="*60)

    if not verify_authentication(args.token):
        return

    # Initialize API
    api = HfApi()

    # Create repository if requested
    if args.create_repo and not args.dry_run:
        print(f"\nCreating repository: {args.repo_id}")
        try:
            api.create_repo(
                repo_id=args.repo_id,
                repo_type="dataset",
                exist_ok=True
            )
            print(f"✓ Repository created/verified: {args.repo_id}")
        except Exception as e:
            print(f"✗ Failed to create repository: {e}")
            return

    # Upload files
    print("\n" + "="*60)
    print("Upload")
    print("="*60)
    print(f"Repository: {args.repo_id}")

    try:
        upload_files(workspace, args.repo_id, api, args.dry_run)
    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        return

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    if args.dry_run:
        print("✓ Dry run completed - no files uploaded")
        print(f"\nTo upload for real, run without --dry-run:")
        print(f"  python 05_upload_to_hf.py --workspace {args.workspace} --repo-id {args.repo_id}")
    else:
        print("✓ Upload completed successfully!")
        print(f"\nDataset available at:")
        print(f"  https://huggingface.co/datasets/{args.repo_id}")
        print(f"\nUsage:")
        print(f"  from datasets import load_dataset")
        print(f"  # Load metadata")
        print(f"  metadata = load_dataset('{args.repo_id}', 'metadata')")
        print(f"  # Load audio")
        print(f"  audio = load_dataset('{args.repo_id}', 'audio')")


if __name__ == '__main__':
    main()
