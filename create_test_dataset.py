#!/usr/bin/env python3
"""
Script to create a test dataset by randomly sampling elements from each source.
Uses only high school sources and keeps alignment across Nemeth, UEB, and MathML versions.
"""

import random
from pathlib import Path


def get_source_files(base_path):
    """Get all deduplicated source files from the highschool directory."""
    sources = []
    for category in ['highschool']:
        category_path = base_path / category
        if category_path.exists():
            for file in category_path.glob('*-no-dups.*'):
                sources.append((category, file.stem.replace('-no-dups', ''), file.suffix))
    return sources


def read_file_lines(file_path):
    """Read all lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()


def sample_aligned_data(nemeth_file, ueb_file, mathml_file, sample_size):
    """
    Sample aligned data from all three file types.
    Returns the sampled lines from each file, maintaining alignment.
    """
    # Read all files
    nemeth_lines = read_file_lines(nemeth_file)
    ueb_lines = read_file_lines(ueb_file)
    mathml_lines = read_file_lines(mathml_file)

    # Verify all files have the same number of lines
    min_lines = min(len(nemeth_lines), len(ueb_lines), len(mathml_lines))

    if min_lines == 0:
        return [], [], []

    # Adjust sample size if file is smaller
    actual_sample_size = min(sample_size, min_lines)

    # Randomly sample line indices
    sampled_indices = sorted(random.sample(range(min_lines), actual_sample_size))

    # Extract sampled lines
    nemeth_samples = [nemeth_lines[i] for i in sampled_indices]
    ueb_samples = [ueb_lines[i] for i in sampled_indices]
    mathml_samples = [mathml_lines[i] for i in sampled_indices]

    return nemeth_samples, ueb_samples, mathml_samples


def write_samples(output_dir, subfolder, source_name, extension, lines):
    """Write sampled lines to output file."""
    output_subdir = output_dir / subfolder
    output_subdir.mkdir(parents=True, exist_ok=True)
    output_file = output_subdir / f"{source_name}{extension}"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def main():
    # Set random seed for reproducibility
    random.seed(42)

    # Define paths
    project_root = Path(__file__).parent
    nemeth_base = project_root / 'BrailleData' / 'Braille' / 'Nemeth'
    ueb_base = project_root / 'BrailleData' / 'Braille' / 'UEB'
    mathml_base = project_root / 'SimpleSpeakData'
    output_dir = project_root / 'testdata'

    # Files to exclude from the dataset
    excluded_sources = {
        'Adventures of Small Number__58__ A collection of short stories',
        'The Story of 8',
    }

    # Get all source files from Nemeth directory (use as reference)
    sources = get_source_files(nemeth_base)

    print(f"Found {len(sources)} source files")

    total_nemeth = 0
    total_ueb = 0
    total_mathml = 0
    processed = 0

    for category, source_name, _ in sources:
        # Skip excluded sources
        if source_name in excluded_sources:
            print(f"Skipping {source_name}: excluded")
            continue

        # Construct file paths
        nemeth_file = nemeth_base / category / f"{source_name}-no-dups.brls"
        ueb_file = ueb_base / category / f"{source_name}-no-dups.brls"
        mathml_file = mathml_base / category / f"{source_name}-no-dups.mmls"

        # Check if all three files exist
        if not (nemeth_file.exists() and ueb_file.exists() and mathml_file.exists()):
            print(f"Skipping {source_name}: missing files")
            continue

        # Sample aligned data
        nemeth_samples, ueb_samples, mathml_samples = sample_aligned_data(
            nemeth_file, ueb_file, mathml_file, sample_size=110
        )

        if not nemeth_samples:
            print(f"Skipping {source_name}: empty file")
            continue

        # Write samples to output directory
        write_samples(output_dir, 'nemeth', source_name, '.brls', nemeth_samples)
        write_samples(output_dir, 'ueb', source_name, '.brls', ueb_samples)
        write_samples(output_dir, 'mathml', source_name, '.mmls', mathml_samples)

        total_nemeth += len(nemeth_samples)
        total_ueb += len(ueb_samples)
        total_mathml += len(mathml_samples)
        processed += 1

        if processed % 10 == 0:
            print(f"Processed {processed} sources...")

    print(f"\nCompleted!")
    print(f"Processed {processed} sources")
    print(f"Total samples:")
    print(f"  - Nemeth: {total_nemeth}")
    print(f"  - UEB: {total_ueb}")
    print(f"  - MathML: {total_mathml}")
    print(f"\nTest dataset saved to: {output_dir}")


if __name__ == '__main__':
    main()
