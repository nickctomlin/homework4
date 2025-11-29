"""
Setup script to prepare training data for the VLM and CLIP models.

This script:
1. Copies images from supertux_data/data/train to data/train
2. Generates QA pairs for VLM training
3. Generates captions for CLIP training

Usage:
    python -m homework.setup_data
"""

import shutil
from pathlib import Path

import fire


def setup_training_data(
    source_dir: str = "supertux_data/data/train",
    target_dir: str = "data/train",
    skip_copy: bool = False,
):
    """
    Set up training data by copying images and generating QA pairs/captions.

    Args:
        source_dir: Source directory containing supertux data
        target_dir: Target directory for training data
        skip_copy: If True, skip copying images (useful if already copied)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist!")
        print("Please download and unzip supertux_data.zip first:")
        print("  wget https://utexas.box.com/shared/static/qubjm5isldqvyimfj9rsmbnvnbezwcv4.zip -O supertux_data.zip")
        print("  unzip supertux_data.zip")
        return

    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Copy images and info files
    if not skip_copy:
        print(f"Copying files from {source_dir} to {target_dir}...")
        files = list(source_path.glob("*"))
        for i, f in enumerate(files):
            if f.is_file():
                shutil.copy2(f, target_path / f.name)
            if (i + 1) % 1000 == 0:
                print(f"  Copied {i + 1}/{len(files)} files...")
        print(f"  Copied {len(files)} files.")
    else:
        print("Skipping image copy (--skip_copy=True)")

    # Step 2: Generate QA pairs
    print("\nGenerating QA pairs...")
    from .generate_qa import generate_all_qa_pairs

    generate_all_qa_pairs(
        data_dir=str(target_path),
        output_file=str(target_path / "balanced_qa_pairs.json"),
        image_prefix="train",
    )

    # Step 3: Generate captions
    print("\nGenerating captions...")
    from .generate_captions import generate_all_captions

    generate_all_captions(
        data_dir=str(target_path),
        output_file=str(target_path / "all_captions.json"),
        image_prefix="train",
    )

    print("\n" + "=" * 60)
    print("Data setup complete!")
    print("=" * 60)
    print("\nYou can now train the VLM model:")
    print("  python -m homework.finetune train")
    print("\nAnd train the CLIP model:")
    print("  python -m homework.clip train")


def main():
    fire.Fire(setup_training_data)


if __name__ == "__main__":
    main()

