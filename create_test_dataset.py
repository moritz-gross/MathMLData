#!/usr/bin/env python3
"""
Script to create test and example datasets by randomly sampling elements each
from high school sources, keeping alignment across Nemeth, UEB, and MathML versions.
The test and example datasets are guaranteed to be disjoint per source file.
"""

import random
from pathlib import Path
from compare_mathml_in_csv import setMathCATPreferences, setMathMLForMathCAT


def get_source_files(base_path):
    """Get all deduplicated source files from the highschool directory."""
    sources = []
    for category in ['highschool']:  # could also add "college" if desired
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


def split_aligned_data(nemeth_file, ueb_file, mathml_file, sample_size, duplicate_oversampling_factor=2):
    """
    Sample two disjoint aligned sets from all three file types.
    Returns (test_samples, example_samples) for each file type.
    """
    nemeth_lines = read_file_lines(nemeth_file)
    ueb_lines = read_file_lines(ueb_file)
    mathml_lines = read_file_lines(mathml_file)

    if len(nemeth_lines) != len(ueb_lines) or len(ueb_lines) != len(mathml_lines):
        print(f"Error: {nemeth_file}[{len(nemeth_lines)}] or "
              f"{ueb_file}[{len(ueb_lines)}] or "
              f"{mathml_file}[{len(mathml_lines)}] has different lengths")
        return ([], [], []), ([], [], [])

    n_lines = len(nemeth_lines)
    test_size = min(2*sample_size, n_lines)

    all_indices = list(range(0, test_size))
    random_indices = random.sample(all_indices, test_size)
    unique_indices: list[int] = []
    unique_lines = set()
    for i in range(0, len(random_indices)):
        if all_indices[i] not in unique_lines:
            unique_lines.add(all_indices[i])
            unique_indices.append(random_indices[i])
    if len(unique_indices) < sample_size:
        return split_aligned_data(nemeth_file, ueb_file, mathml_file, sample_size, 2*duplicate_oversampling_factor)

    unique_indices = unique_indices[:2*sample_size]
    unique_indices = sorted(unique_indices)
    test_nemeth = []
    test_ueb = []
    test_mathml = []
    example_nemeth = []
    example_ueb = []
    example_mathml = []
    for i in range(0, len(unique_indices), 2):
        test_nemeth.append(nemeth_lines[unique_indices[i]])
        test_ueb.append(ueb_lines[unique_indices[i]])
        test_mathml.append(mathml_lines[unique_indices[i]])

        example_nemeth.append(nemeth_lines[unique_indices[i+1]])
        example_ueb.append(ueb_lines[unique_indices[i+1]])
        example_mathml.append(mathml_lines[unique_indices[i+1]])

    return (test_nemeth, test_ueb, test_mathml), (example_nemeth, example_ueb, example_mathml)


def canonicalize_mathml_list(mathml_list: list[str]) -> list[str]:
    """Canonicalize a list of MathML strings using libmathcat.setMathMLForMathCAT.
    Returns a list with one MathML expression per line (newlines in MathML replaced with spaces).
    """
    canonicalized = []
    for mathml in mathml_list:
        mathml_stripped = mathml.strip()
        if not mathml_stripped:
            canonicalized.append('\n')
            continue
        try:
            canonical = setMathMLForMathCAT(mathml_stripped)
            canonicalized.append(canonical + '\n')
        except Exception as e:
            print(f"Warning: Error canonicalizing MathML: {e}")
            canonicalized.append('\n')
    return canonicalized


def main():
    random.seed(42)

    # Initialize MathCAT
    setMathCATPreferences({})

    project_root = Path(__file__).parent
    nemeth_base = project_root / 'BrailleData' / 'Braille' / 'Nemeth'
    ueb_base = project_root / 'BrailleData' / 'Braille' / 'UEB'
    mathml_base = project_root / 'SimpleSpeakData'
    test_output_dir = project_root / 'test_data'
    example_output_dir = project_root / 'example_data'
    test_canonical_output_dir = test_output_dir / 'CanonicalMathML'
    example_canonical_output_dir = example_output_dir / 'CanonicalMathML'

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

    all_test_nemeth, all_test_ueb, all_test_mathml = [], [], []
    all_example_nemeth, all_example_ueb, all_example_mathml = [], [], []
    all_canonical_test_mathml, all_canonical_example_mathml = [], []

    for category, source_name, _ in sources:
        if source_name in excluded_sources:
            print(f"Skipping {source_name}: excluded")
            continue

        nemeth_file = nemeth_base / category / f"{source_name}-no-dups.brls"
        ueb_file = ueb_base / category / f"{source_name}-no-dups.brls"
        mathml_file = mathml_base / category / f"{source_name}-no-dups.mmls"

        # Sample disjoint aligned data
        (test_nemeth, test_ueb, test_mathml), (ex_nemeth, ex_ueb, ex_mathml) = split_aligned_data(
            nemeth_file, ueb_file, mathml_file, sample_size=120
        )

        write_samples(test_output_dir, 'Nemeth', source_name, '.brls', test_nemeth)
        write_samples(test_output_dir, 'UEB', source_name, '.brls', test_ueb)
        write_samples(test_output_dir, 'MathML', source_name, '.mmls', test_mathml)

        write_samples(example_output_dir, 'Nemeth', source_name, '.brls', ex_nemeth)
        write_samples(example_output_dir, 'UEB', source_name, '.brls', ex_ueb)
        write_samples(example_output_dir, 'MathML', source_name, '.mmls', ex_mathml)

        # Canonicalize test and example MathML entries
        canonical_test_mathml = canonicalize_mathml_list(test_mathml)
        canonical_example_mathml = canonicalize_mathml_list(ex_mathml)
        write_samples(test_canonical_output_dir, '', source_name, '.mmls', canonical_test_mathml)
        write_samples(example_canonical_output_dir, '', source_name, '.mmls', canonical_example_mathml)

        all_test_nemeth.extend(canonical_test_mathml)
        all_test_ueb.extend(test_ueb)
        all_test_mathml.extend(test_mathml)
        all_example_nemeth.extend(ex_nemeth)
        all_example_ueb.extend(ex_ueb)
        all_example_mathml.extend(ex_mathml)
        all_canonical_test_mathml.extend(canonical_test_mathml)
        all_canonical_example_mathml.extend(canonical_example_mathml)

    random.shuffle(all_test_nemeth)
    random.shuffle(all_test_ueb)
    random.shuffle(all_test_mathml)
    random.shuffle(all_example_nemeth)
    random.shuffle(all_example_ueb)
    random.shuffle(all_example_mathml)
    random.shuffle(all_canonical_test_mathml)
    random.shuffle(all_canonical_example_mathml)

    with open(f"{test_output_dir}/nemeth.brls", 'w', encoding='utf-8') as f:
        f.writelines(all_test_nemeth)
    with open(f"{test_output_dir}/ueb.brls", 'w', encoding='utf-8') as f:
        f.writelines(all_test_ueb)
    with open(f"{test_output_dir}/mathml.mmls", 'w', encoding='utf-8') as f:
        f.writelines(all_test_mathml)
    with open(f"{example_output_dir}/nemeth.brls", 'w', encoding='utf-8') as f:
        f.writelines(all_example_nemeth)
    with open(f"{example_output_dir}/ueb.brls", 'w', encoding='utf-8') as f:
        f.writelines(all_example_ueb)
    with open(f"{example_output_dir}/mathml.mmls", 'w', encoding='utf-8') as f:
        f.writelines(all_example_mathml)
    with open(f"{test_output_dir}/canonical-mathml.mmls", 'w', encoding='utf-8') as f:
        f.writelines(all_canonical_test_mathml)
    with open(f"{example_output_dir}/canonical-mathml.mmls", 'w', encoding='utf-8') as f:
        f.writelines(all_canonical_example_mathml)

    print("\nCompleted!")
    print(f"Processed {processed} sources")
    print("Total samples:")
    print(f"  - Test Nemeth: {total_test_nemeth}")
    print(f"  - Test UEB: {total_test_ueb}")
    print(f"  - Test MathML: {total_test_mathml}")
    print(f"  - Example Nemeth: {total_example_nemeth}")
    print(f"  - Example UEB: {total_example_ueb}")
    print(f"  - Example MathML: {total_example_mathml}")
    print(f"\nTest dataset saved to: {test_output_dir}")
    print(f"Example dataset saved to: {example_output_dir}")
    print(f"Test Canonical MathML dataset saved to: {test_canonical_output_dir}")
    print(f"Example Canonical MathML dataset saved to: {example_canonical_output_dir}")


if __name__ == '__main__':
    main()
