import xml.etree.ElementTree as ET
import yaml
from typing import Any
import sys
sys.stdout.reconfigure(encoding='utf-8')  # in case print statements are used for debugging


def get_unique_mathml_chars(file_path):
    unique_chars = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                # Parse the MathML line
                # Note: This assumes each line is a valid XML fragment (e.g., starts with <math>)
                root = ET.fromstring(line)

                # itertext() extracts all text between tags and automatically
                # converts entities like &#x2211; to their Unicode equivalents
                for text_content in root.itertext():
                    unique_chars.update(text_content)

            except ET.ParseError:
                # If a line isn't a complete XML tree, you might need to wrap it
                # root = ET.fromstring(f"<root>{line}</root>")
                continue

    return unique_chars


# Python 3.12+ Type Alias
type CharMapping = dict[str, str]
type CharSet = set[str]

UEB_REPLACEMENT_CHARS: CharMapping = {
    "S": "XXX",
    "B": "⠘",
    "𝔹": "XXX",
    "T": "⠈",
    "I": "⠨",
    "R": "",
    "1": "⠰",
    "𝟙": "⠰⠰",
    "L": "",
    "D": "XXX",
    "G": "",
    "V": "⠨⠈",
    "C": "⠠",
    "𝐶": "⠠",
    "N": "⠼",
    "t": "⠱",
    "W": "⠀",
    "𝐖": "⠀",
    "s": "⠆",
    "w": "⠂",
    "e": "⠄",
    "o": "",
    "c": "",
    "b": "",
    ",": "⠂",
    ".": "⠲",
    "-": "-",
    "—": "⠠⠤",
    "―": "⠐⠠⠤",
    "#": "",
}

NEMETH_REPLACEMENT_CHARS: dict[str, str] = {
    "S": "⠠⠨",
    "B": "⠸",
    "𝔹": "⠨",
    "T": "⠈",
    "I": "⠨",
    "R": "",
    "E": "⠰",
    "D": "⠸",
    "G": "⠨",
    "V": "⠨⠈",
    "H": "⠠⠠",
    "U": "⠈⠈",
    "C": "⠠",
    "P": "⠸",
    "𝐏": "⠸",
    "L": "",
    "l": "",
    "M": "",
    "m": "⠐",
    "N": "",
    "n": "⠼",
    "𝑁": "",
    "W": "⠀",
    "w": "⠀",
    ",": "⠠⠀",
    "b": "⠐",
    "𝑏": "⣐",
    "↑": "⠘",
    "↓": "⠰",
}


def generate_braille_mapping(
    yaml_file_path: str,
    indicator_replacements: CharMapping,
    chars_output_file: str,
    char_set: CharSet
) -> None:
    """
    Parses a YAML file and prints a mapping of keys to their 'else' or
    default braille values for all characters present in char_set.
    """
    try:
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            # Type checkers require Any here as YAML structure is dynamic
            data: Any = yaml.safe_load(f)

        if not isinstance(data, list):
            return

        mapping: CharMapping = {}

        for entry in data:
            if not isinstance(entry, dict):
                continue

            for key, value in entry.items():
                match value:
                    # Case 1: Simple list [t: "braille"]
                    case [{"t": str(b)}]:
                        mapping[key] = b

                    # Case 2: Nested test structure with 'else' branch
                    case [{"test": {"else": [{"t": str(b)}]}}]:
                        mapping[key] = b

                    case _:
                        continue
                # replace the indicators with the replacement characters
                if key == " ":
                    del mapping[key]
                    key = " "
                    mapping[key] = "⠀"      # non breaking space -> empty braille dots
                mapping[key] = "".join(indicator_replacements.get(ch, ch) for ch in mapping[key])

        # Sort for deterministic output; prints only keys found in the YAML
        with open(chars_output_file, "w", encoding="utf-8") as f:
            for char in sorted(char_set):
                if braille := mapping.get(char):
                    f.write(f"{char} | {braille}\n")

    except FileNotFoundError:
        print(f"Error: {yaml_file_path} not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Example usage:
# char_set: CharSet = {'≠', '≡', '≤'}
# generate_braille_mapping('symbols.yaml', char_set)
def main():
    unicode_file_name = sys.argv[1]
    chars_output_file = sys.argv[2]
    test_unique_chars = get_unique_mathml_chars("test_data/mathml.mmls")
    example_unique_chars = get_unique_mathml_chars("example_data/mathml.mmls")
    all_unique_chars = test_unique_chars | example_unique_chars
    print(f"Unique characters: {len(all_unique_chars)}; "
          f"example: {len(example_unique_chars)}; "
          f"test: {len(test_unique_chars)}")
    print(sorted(list(all_unique_chars)))
    if "nemeth" in unicode_file_name.lower():
        indicator_dict = NEMETH_REPLACEMENT_CHARS
    else:
        indicator_dict = UEB_REPLACEMENT_CHARS
    generate_braille_mapping(unicode_file_name, indicator_dict, chars_output_file, all_unique_chars)

# Example usage:
# char_set = get_unique_mathml_chars('data.mml')
# print(f"Unique characters: {sorted(list(char_set))}")


if __name__ == '__main__':
    main()
