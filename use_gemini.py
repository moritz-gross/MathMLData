import os
import sys
from compare_mathml_in_csv import setMathCATPreferences, areCanonicallyEqual, CanonicalResults
import itertools
from google import genai
from google.genai import types
import json
import time
import re
sys.stdout.reconfigure(encoding='utf-8')   # Ensure UTF-8 output for Unicode Braille


def convert_ueb_unicode_to_mathml(instructions: str, braille_input: list[str], model: str) -> str:
    """
    Uses Gemini 3 Flash Preview to convert a Unicode UEB Braille string to MathML.
    """

    # 1. Setup Client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable.")

    client = genai.Client(api_key=api_key, http_options={"timeout": 2400000})  # isn't allowed in generate_content

    try:
        start_time = time.time()
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            # model="gemini-3-pro-preview",
            config=types.GenerateContentConfig(
                system_instruction=instructions,
                temperature=0.1,  # Low temperature for precision
                response_mime_type="application/json",  # Crucial: Forces JSON output
                response_schema=list[str]               # Crucial: Defines the shape
            ),
            contents=str(braille_input),
        )

        end_time = time.time()
        print(f"Latency: {end_time - start_time} seconds")
        print("Token Usage:")
        if response.usage_metadata:
            print(f"   • Prompt Tokens: {response.usage_metadata.prompt_token_count}")
            print(f"   • Response Tokens: {response.usage_metadata.candidates_token_count}")
            print(f"   • Total Tokens: {response.usage_metadata.total_token_count}")
        else:
            print("   No usage metadata returned.")

        print(f"response.text='...{response.text[len(response.text)-900:]}'")  # Print last 900 characters of response for debugging
        json_str = response.text
        as_list = json.loads(json_str)
        print(f"last five items of as_list=...\n{'\n'.join(as_list[len(as_list)-5:])}\n")  # Print last 5 items for debugging
        return as_list
    except Exception as e:
        # Try to parse the response text if possible (sometimes the string is truncated)
        pattern = r'"((?:[^"\\]|\\.)*)"'    # Regex to match strings
        as_list = re.findall(pattern, json_str)
        print(f"\n***Error during conversion: {e}. Partial {len(as_list)} results extracted out of {len(braille_input)} tests\n")
        return as_list


def write_results_to_file(braille_tests: list[str], computed_output: list[str], expected_output: list[str], output_file: str) -> None:
    """
    Write the results out after comparing the computed and expected MathML outputs.
    """
    if not isinstance(computed_output, list) or not computed_output[0].startswith('<math'):
        print(f"Error: Computed output is a {type(computed_output)},\
              {len(computed_output) if isinstance(computed_output, list) else 0} items")
        if isinstance(computed_output, list):
            print("Computed output first 5 lines:\n", computed_output[:5])
        return

    # initial MathCAT
    setMathCATPreferences({})

    with open(output_file, "w", encoding="utf-8") as f:
        match_count = 0
        f.write("\nNOT Normalized MathML\n")
        f.write("Test Braille | Match | Expected MathML | Computed MathML\n")
        for tests, computed, expected in itertools.zip_longest(braille_tests, computed_output, expected_output, fillvalue=""):
            try:
                checked = areCanonicallyEqual(expected, computed)
                if checked.isEqual:
                    match_count += 1
            except:
                checked = CanonicalResults(False, "", "")
            match = "✓" if checked.isEqual else "✗"
            f.write(f"{match} | {tests} | {expected} | {computed}\n")

        f.write("\nNormalized MathML\n")
        f.write("Test Braille | Match | Expected MathML | Computed MathML\n")
        for tests, computed, expected in itertools.zip_longest(braille_tests, computed_output, expected_output, fillvalue=""):
            try:
                checked = areCanonicallyEqual(expected, computed)
            except Exception as e:
                print(f"areCanonicallyEqual error message:\n{e}", file=sys.stderr)
                checked = CanonicalResults(False, expected, '<--bad MathML-->' + computed)

            match = "✓" if checked.isEqual else "✗"
            f.write(f"{match} | {tests} | {checked.canonicalOriginal} | {checked.canonicalComputed}\n")

        print(f"Results written to {output_file}. Total Matches: {match_count} out of {len(braille_tests)}: {match_count/len(braille_tests)*100:.2f}%.")


def generateExamples(braille_path, mathml_path) -> str:
    """
    Zips lines from two files and returns a string of the from "braille | mathml\n".
    """
    try:
        with open(braille_path, "r", encoding="utf-8") as f:
            braille = f.read().splitlines()
        with open(mathml_path, "r", encoding="utf-8") as f:
            mathml = f.read().splitlines()

        # Zip and print
        result = ""
        for brl, mml in zip(braille, mathml):
            # .strip() removes the trailing newline character for cleaner output
            quoted_mml = mml.strip().replace('"', '\\"')
            result += f'"{brl.strip()} | {quoted_mml}\\n"\n'

        return result

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e.filename}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Example: E = mc^2 in UEB
    # Breakdown:
    # ⠠⠑  -> Capital E
    # ⠀    -> Space
    # ⠐⠶  -> Equals sign
    # ⠀    -> Space
    # ⠍    -> m
    # ⠉    -> c
    # ⠔⠼⠃ -> Superscript 2

    instructionsProlog = (
        "You are an expert Braille translator specializing in Nemeth braille. "
        "The user will provide a python list of strings, "
        "where each string is composed of Unicode Nemeth Braille characters (e.g., ['⠠⠑⠀⠨⠅⠀⠍⠉⠘⠆', '⠎⠊⠝⠀⠷⠨⠹⠾']). "
        "Your task is to translate each exact Braille sequence of characters into valid MathML code. "
        "For each braille input, output ONLY the raw MathML string starting with <math> and ending with </math>. "
        "Every element in the MathML must be properly closed and nested. "
        "Do not include markdown formatting, explanations, or any other text."
        "Return the results as a python list of strings, where each string is the MathML translation of the corresponding braille input. "
        "Below are some examples of braille/MathML pairs separated by '|' that should be considered the ground truth:\n"
    )

    examples = generateExamples("RustTestData/Nemeth.brls", "RustTestData/Nemeth.mmls")

    instructions = instructionsProlog + examples
    # unicode = read_unicode("xxx")
    with open("BrailleData/Braille/Nemeth/highschool/Algebra Toolkit-no-dups.brls", "r", encoding="utf-8") as f:
        braille = f.read().splitlines()
    with open("SimpleSpeakData/highschool/Algebra Toolkit-no-dups.mmls", "r", encoding="utf-8") as f:
        mathml = f.read().splitlines()
    if len(braille) != len(mathml):
        print("Error: Number of test inputs does not match number of expected outputs.")
        sys.exit(1)

    braille = braille[220:420]
    mathml = mathml[220:420]
    # model = "gemini-3-flash-preview"
    model = "gemini-3-pro-preview"
    brailleCode = "Nemeth"
    n_examples = examples.count('\n')
    print(f"Running test with {n_examples} examples, {len(braille)} tests with {model} for {brailleCode}.")
    mathml_output = convert_ueb_unicode_to_mathml(instructions, braille, model)
    write_results_to_file(braille, mathml_output, mathml, f"{brailleCode}-{model}-{n_examples}exs-{len(braille)}tsts.txt")
