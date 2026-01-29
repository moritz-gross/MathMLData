import re
import sys
import os
from typing import Tuple, List
from compare_mathml_in_csv import setMathCATPreferences, setMathMLForMathCAT
sys.stdout.reconfigure(encoding='utf-8')  # in case print statements are used for debugging


def extract_from_file(filename: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract expr (MathML) strings, braille strings, and canonical MathML from a Rust test file.

    Args:
        filename: Path to the Rust test file

    Returns:
        tuple: (expr_list, braille_list, canonical_list) - Three lists containing the extracted strings
    """
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to strip // comments but not if they are inside strings
    comment_pattern = re.compile(
        r'('                       # Start Group 1 (Stuff to keep)
            r'r(#*)"[\s\S]*?"\2'   # Raw strings: r#""# or r""
            r'|'                   # OR
            r'"(?:\\.|[^"\\])*"'   # Standard strings: "..."
        r')'                       # End Group 1
        r'|'                       # OR (Stuff to delete)
        r'//.*',                   # The comment
        re.MULTILINE
    )

    # Strip comments from content
    content = comment_pattern.sub(lambda m: m.group(1) or "", content)

    # Use combined pattern to ensure expr and test_braille are paired correctly
    # Pattern for "expr = ..."
    expr_part = r'let\s+expr\s*=\s*(?:r#*"(.*?)"#*|"(.*?)")\s*;'
    # Pattern for "test_braille(...)"
    call_part = r'\s*test_braille(?:_prefs)?\s*\(\s*.*?\s*expr\s*,\s*"([^"]*)"\s*\)\s*;'
    # Combined pattern matches both together
    combined_pattern = re.compile(expr_part + call_part, re.MULTILINE | re.DOTALL)

    # Extract paired matches
    matches = combined_pattern.findall(content)

    expr_list = []
    braille_list = []
    canonical_list = []

    for i, match in enumerate(matches):
        # match[0] is the content for raw strings (r#"..."#)
        # match[1] is the content for standard strings ("...")
        # match[2] is the braille string
        expr_content = match[0] if match[0] else match[1]
        braille_content = match[2]

        # Normalize whitespace but preserve structure
        expr_content = " ".join(expr_content.split()).strip()
        expr_list.append(expr_content)
        braille_list.append(braille_content)

        # Generate canonical MathML
        if expr_content.startswith("<math") and expr_content.endswith("</math>"):
            try:
                canonical = setMathMLForMathCAT(expr_content)
                canonical_list.append(" ".join(canonical.split()).strip())
            except Exception as e:
                print(f"Warning: Canonicalization error for test in {filename}: {e}")
                print(f"MathML={expr_content[:100]}...")
                canonical_list.append("")  # Add empty string to keep lists aligned
        else:
            canonical_list.append("")  # Add empty string if not valid MathML

    return expr_list, braille_list, canonical_list


def extract_from_files(file_list: List[str], expr_output: str, braille_output: str, canonical_output: str) -> None:
    """
    Extract expr, braille strings, and canonical MathML from multiple Rust test files and write to output files.

    Args:
        file_list: List of paths to Rust test files
        expr_output: Path to output file for expr strings
        braille_output: Path to output file for braille strings
        canonical_output: Path to output file for canonical MathML
    """
    # Clear existing output files if they exist
    output_files = [expr_output, braille_output, canonical_output]

    for out_file in output_files:
        if os.path.exists(out_file):
            os.remove(out_file)

    # Initialize MathCAT
    try:
        setMathCATPreferences({})
    except Exception as e:
        print(f"Warning: Can't set MathCAT preferences: {e}")

    total_expr_count = 0
    total_braille_count = 0
    total_canonical_count = 0

    for filename in file_list:
        if not os.path.isfile(filename):
            print(f"Skipping: '{filename}' (File not found)")
            continue

        expr_list, braille_list, canonical_list = extract_from_file(filename)

        # Append to output files
        with open(expr_output, 'a', encoding='utf-8') as f:
            for expr in expr_list:
                f.write(f"{expr}\n")

        with open(braille_output, 'a', encoding='utf-8') as f:
            for braille in braille_list:
                f.write(f"{braille}\n")

        with open(canonical_output, 'a', encoding='utf-8') as f:
            for canonical in canonical_list:
                f.write(f"{canonical}\n")

        total_expr_count += len(expr_list)
        total_braille_count += len(braille_list)
        total_canonical_count += len(canonical_list)
        print(f"Processed {filename}: Found {len(expr_list)} pairs.")

    print(f"\nTotal extracted: {total_expr_count} expr strings, "
          f"{total_braille_count} braille strings, "
          f"and {total_canonical_count} canonical MathML strings")
    print(f"Written to {expr_output}, {braille_output}, and {canonical_output}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python extract_rust_test_data.py "
              "<expr_output_file> "
              "<braille_output_file> "
              "<canonical_output_file> "
              "<input_file1> [input_file2] ...")
        sys.exit(1)

    expr_output = sys.argv[1]
    braille_output = sys.argv[2]
    canonical_output = sys.argv[3]
    input_files = sys.argv[4:]

    extract_from_files(input_files, expr_output, braille_output, canonical_output)
