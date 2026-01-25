#!/usr/bin/env python3
"""
Script to create test and example datasets by randomly sampling elements each
from high school sources, keeping alignment across Nemeth, UEB, and MathML versions.
The test and example datasets are guaranteed to be disjoint per source file.
"""

import random
from pathlib import Path


def get_source_files(base_path):
    """Get all deduplicated source files from the highschool directory."""
    sources = []
    for category in ['highschool']: # could also add "college" if desired
        category_path = base_path / category
        if category_path.exists():
            for file in category_path.glob('*-no-dups.*'):
                sources.append((category, file.stem.replace('-no-dups', ''), file.suffix))
    return sources


def read_file_lines(file_path):
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


def split_aligned_data(nemeth_file, ueb_file, mathml_file, sample_size):
    """
    Sample two disjoint aligned sets from all three file types.
    Returns (test_samples, example_samples) for each file type.
    """
    nemeth_lines = read_file_lines(nemeth_file)
    ueb_lines = read_file_lines(ueb_file)
    mathml_lines = read_file_lines(mathml_file)

    min_lines = min(len(nemeth_lines), len(ueb_lines), len(mathml_lines))
    if min_lines == 0:
        return ([], [], []), ([], [], [])

    test_size = min(sample_size, min_lines)
    remaining = min_lines - test_size
    example_size = min(sample_size, remaining)

    all_indices = list(range(min_lines))
    test_indices = set(random.sample(all_indices, test_size))
    remaining_indices = [i for i in all_indices if i not in test_indices]
    example_indices = set(random.sample(remaining_indices, example_size))

    test_indices_sorted = sorted(test_indices)
    example_indices_sorted = sorted(example_indices)

    test_nemeth = [nemeth_lines[i] for i in test_indices_sorted]
    test_ueb = [ueb_lines[i] for i in test_indices_sorted]
    test_mathml = [mathml_lines[i] for i in test_indices_sorted]

    example_nemeth = [nemeth_lines[i] for i in example_indices_sorted]
    example_ueb = [ueb_lines[i] for i in example_indices_sorted]
    example_mathml = [mathml_lines[i] for i in example_indices_sorted]

    return (test_nemeth, test_ueb, test_mathml), (example_nemeth, example_ueb, example_mathml)


def main():
    random.seed(42)

    project_root = Path(__file__).parent
    nemeth_base = project_root / 'BrailleData' / 'Braille' / 'Nemeth'
    ueb_base = project_root / 'BrailleData' / 'Braille' / 'UEB'
    mathml_base = project_root / 'SimpleSpeakData'
    test_output_dir = project_root / 'test_data'
    example_output_dir = project_root / 'example_data'

    excluded_sources = {
        'Adventures of Small Number__58__ A collection of short stories',
        'The Story of 8',
    }

    # Get all source files from Nemeth directory (use as reference)
    sources = get_source_files(nemeth_base)

    print(f"Found {len(sources)} source files")

    total_test_nemeth = 0
    total_test_ueb = 0
    total_test_mathml = 0
    total_example_nemeth = 0
    total_example_ueb = 0
    total_example_mathml = 0
    processed = 0

    for category, source_name, _ in sources:
        if source_name in excluded_sources:
            print(f"Skipping {source_name}: excluded")
            continue

        nemeth_file = nemeth_base / category / f"{source_name}-no-dups.brls"
        ueb_file = ueb_base / category / f"{source_name}-no-dups.brls"
        mathml_file = mathml_base / category / f"{source_name}-no-dups.mmls"

        # Sample disjoint aligned data
        (test_nemeth, test_ueb, test_mathml), (ex_nemeth, ex_ueb, ex_mathml) = split_aligned_data(
            nemeth_file, ueb_file, mathml_file, sample_size=110
        )

        write_samples(test_output_dir, 'nemeth', source_name, '.brls', test_nemeth)
        write_samples(test_output_dir, 'ueb', source_name, '.brls', test_ueb)
        write_samples(test_output_dir, 'mathml', source_name, '.mmls', test_mathml)

        write_samples(example_output_dir, 'nemeth', source_name, '.brls', ex_nemeth)
        write_samples(example_output_dir, 'ueb', source_name, '.brls', ex_ueb)
        write_samples(example_output_dir, 'mathml', source_name, '.mmls', ex_mathml)

        total_test_nemeth += len(test_nemeth)
        total_test_ueb += len(test_ueb)
        total_test_mathml += len(test_mathml)
        total_example_nemeth += len(ex_nemeth)
        total_example_ueb += len(ex_ueb)
        total_example_mathml += len(ex_mathml)
        processed += 1

    print(f"\nCompleted!")
    print(f"Processed {processed} sources")
    print(f"Total samples:")
    print(f"  - Test Nemeth: {total_test_nemeth}")
    print(f"  - Test UEB: {total_test_ueb}")
    print(f"  - Test MathML: {total_test_mathml}")
    print(f"  - Example Nemeth: {total_example_nemeth}")
    print(f"  - Example UEB: {total_example_ueb}")
    print(f"  - Example MathML: {total_example_mathml}")
    print(f"\nTest dataset saved to: {test_output_dir}")
    print(f"Example dataset saved to: {example_output_dir}")


if __name__ == '__main__':
    main()
