import pathlib
import sys
from tqdm import tqdm

def print_usage():
    """Displays instructions on how to run the script."""
    usage = """
Python MMLS Deduplicator
------------------------
Usage:
    python script_name.py [directory_path]

Arguments:
    directory_path    The path to the folder containing .mmls files.
    --help            Show this help message.

Description:
    Recursively finds all .mmls files, removes duplicate lines (ignoring 
    trailing whitespace), and saves a new file with the '-no-dups' suffix.
    """
    print(usage)

def process_mmls_files(root_dir):
    path = pathlib.Path(root_dir)
    
    if not path.is_dir():
        print(f"Error: '{root_dir}' is not a valid directory.")
        return

    # Find all .mmls files recursively
    files = [f for f in path.rglob('*.mmls') if f.is_file()]
    
    if not files:
        print("No .mmls files found in the specified directory.")
        return

    stats = {'files_created': 0, 'lines_removed': 0, 'errors': 0}

    for file_path in tqdm(files, desc="Processing", unit="file"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()
            
            # Remove trailing whitespace for comparison but keep newlines for file structure
            cleaned_lines = [line.rstrip() + '\n' for line in original_lines]
            
            # dict.fromkeys preserves order of first appearance
            unique_lines = list(dict.fromkeys(cleaned_lines))
            
            # Construct new filename: filename-no-dups.mmls
            new_filename = f"{file_path.stem}-no-dups{file_path.suffix}"
            output_path = file_path.with_name(new_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(unique_lines)
            
            stats['files_created'] += 1
            stats['lines_removed'] += len(original_lines) - len(unique_lines)
            
        except (UnicodeDecodeError, PermissionError):
            stats['errors'] += 1

    print_summary(stats)

def print_summary(stats):
    print("\n" + "="*35)
    print(f"{'PROCESS COMPLETE':^35}")
    print("="*35)
    print(f"New Files Created:  {stats['files_created']}")
    print(f"Total Lines Removed: {stats['lines_removed']}")
    if stats['errors'] > 0:
        print(f"Errors encountered:  {stats['errors']}")
    print("="*35)

if __name__ == "__main__":
    # Check if arguments are missing or user asked for help
    if len(sys.argv) != 2 or sys.argv[1] in ['--help', '-h']:
        print_usage()
    else:
        target_dir = sys.argv[1]
        process_mmls_files(target_dir)