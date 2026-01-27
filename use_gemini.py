import time
import logging
import re
import argparse
from typing import Any, NamedTuple  # 'Any' is still imported from typing
from google import genai
from google.genai import types
from google.api_core import exceptions
from compare_mathml_in_csv import setMathCATPreferences, areCanonicallyEqual, CanonicalResults
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')   # Ensure UTF-8 output for Unicode Braille

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class RunConfig(NamedTuple):
    """Configuration values for a Gemini API run."""
    braille_code: str
    gen_braille: bool
    model: str
    apiKeyName: str
    batch_size: int
    n_examples: int
    chunk: slice | None
    instructions: str
    example_braille_file: str
    example_mathml_file: str
    input_braille_file: str
    input_mathml_file: str


def convert_input_with_model(
    instructions: str,
    input: list[str],
    model: str,
    apiKeyName: str,
    batch_size: int = 20
) -> tuple[list[str], dict[str, int], float]:
    """
    Splits input into batches, processes them with retries/streaming,
    tracks token usage, and measures pure generation time.
    """

    # Setup Client
    api_key = os.environ.get(apiKeyName)
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable.")
    client = genai.Client(api_key=api_key, http_options={"timeout": 2400000})  # isn't allowed in generate_content

    # 1. Initialize accumulators
    all_results = ""
    total_tokens: dict[str, int] = {"prompt": 0, "candidates": 0, "total": 0}
    total_generation_time: float = 0.0
    first_attempt = True

    # 2. Loop through the data in chunks
    for i in range(0, len(input), batch_size):
        batch = input[i:i + batch_size]
        batch_id = (i // batch_size) + 1
        print(f"\n--- Processing Batch {batch_id} (Items {i+1} to {i+len(batch)}) ---")

        # 3. Call helper (now returns duration too)
        # We use Any for usage_metadata to be safe across SDK versions
        try:
            batch_text, batch_usage, batch_time = generate_with_retry(client, model, instructions, batch, 3)
        except (exceptions.ServiceUnavailable, exceptions.ServerError) as e:
            if first_attempt:
                # reestablish connection and try one more time
                client = genai.Client(api_key=api_key, http_options={"timeout": 2400000})  # likely large than needed
                first_attempt = False
                batch_text, batch_usage, batch_time = generate_with_retry(client, model, instructions, batch, 3)
            else:
                print(f"Exception raised twice during generation: {e}")
                batch_text, batch_usage, batch_time = None, None, 0.0
                break

        except Exception as e:
            print(f"Exception raised during generation: {e}")
            batch_text, batch_usage, batch_time = None, None, 0.0
            break

        # 4. Process results
        if batch_text:
            all_results += '|next-item|' + batch_text

        # 5. Update stats
        if batch_usage:
            # Assuming batch_usage is a types.GenerateContentResponseUsageMetadata object
            total_tokens["prompt"] += batch_usage.prompt_token_count
            total_tokens["candidates"] += batch_usage.candidates_token_count
            total_tokens["total"] += batch_usage.total_token_count

        if batch_time:
            total_generation_time += batch_time
            print(f"   > Batch Time: {batch_time:.2f}s")
            # We use safe access or Any here since it's a specific SDK object
            if batch_usage:
                print(f"   > Batch Token Usage: {batch_usage.total_token_count} "
                      f"(Prompt: {batch_usage.prompt_token_count}, Output: {batch_usage.candidates_token_count})")

    # trim the start and end, then split the string at '|next-item|' and return a list of strings
    text = all_results
    if input[0].find("<math"):
        i_start = text.find("<math")
        i_end = text.rfind("</math>") + len("</math>")
        if i_start == -1 or i_end == -1:
            raise Exception("Could not find MathML tags in the response.")
    else:
        matches = list(re.finditer(r'[\u2800-\u28ff]', text))
        if not matches:
            raise Exception("Could not find braille chars in the response.")
        i_start = matches[0].start()
        i_end = matches[-1].end()

    # Return the substring including everything between the first and last Braille char
    as_list = text[i_start:i_end].split("|next-item|")
    as_list = [item.strip() for item in as_list if item.strip()]  # Clean up whitespace and remove empty strings
    return as_list, total_tokens, total_generation_time


def _sum_usage(usage1: Any, usage2: Any) -> Any:
    """Helper to sum two UsageMetadata objects."""
    if not usage1:
        return usage2
    if not usage2:
        return usage1

    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=usage1.prompt_token_count + usage2.prompt_token_count,
        candidates_token_count=usage1.candidates_token_count + usage2.candidates_token_count,
        total_token_count=usage1.total_token_count + usage2.total_token_count
    )


def generate_with_retry(
    client: genai.Client,
    model: str,
    instructions: str,
    content: list[str],
    max_retries: int = 3,
    depth: int = 0
) -> tuple[str, Any, float]:

    indent = "  " * depth
    t0 = time.perf_counter()
    delay = 2

    # Attempt Loop (Handles 503s and standard failures)
    for attempt in range(1, max_retries + 1):
        try:
            payload_text = "\n".join(content)

            response_stream = client.models.generate_content_stream(
                model=model,
                config=types.GenerateContentConfig(
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_NONE",
                        )
                    ],
                    system_instruction=instructions,
                    temperature=0.1,
                ),
                contents=payload_text,
            )

            full_text_list: list[str] = []
            first_token_received = False
            final_usage: Any = None
            finish_reason = None

            for chunk in response_stream:
                # Capture the exact moment the first chunk arrives
                if not first_token_received:
                    time_to_first_token = time.perf_counter() - t0
                    print(f"⚡ Time to First Token: {time_to_first_token:.4f} seconds")
                    first_token_received = True
                if chunk.text:
                    full_text_list.append(chunk.text)
                if chunk.usage_metadata:
                    final_usage = chunk.usage_metadata
                if chunk.candidates:
                    finish_reason = chunk.candidates[0].finish_reason

            # --- Validation ---
            if finish_reason == "MAX_TOKENS":
                raise ValueError("MAX_TOKENS")

            if finish_reason != "STOP":
                raise Exception(f"Incomplete generation: {finish_reason}")

            print("\n\n--- Performance Summary ---")
            total_time = time.perf_counter() - t0
            print(f"Total Latency:    {total_time:.2f} s")
            print(f"Time to 1st Token:{time_to_first_token:.2f} s")
            print(f"Generation Time:  {total_time - time_to_first_token:.2f} s (Streaming duration)")
            return "".join(full_text_list), final_usage, total_time - time_to_first_token

        # --- Recursive Split Logic (Divide and Conquer) ---
        except ValueError as e:
            # Only handle MAX_TOKENS here; re-raise actual ValueErrors
            if str(e) != "MAX_TOKENS":
                raise e

            print(f"{indent}[!] MAX_TOKENS hit on {len(content)} lines.")

            if len(content) <= 1:
                print(f"{indent}[X] Critical: Single input line is too large.")
                raise e

            mid = len(content) // 2
            left_part = content[:mid]
            right_part = content[mid:]

            print(f"{indent}    -> Splitting: {len(left_part)} lines | {len(right_part)} lines")

            # Recursive calls (Depth + 1)
            # We don't need to catch exceptions here; let them bubble up
            text_a, usage_a, _ = generate_with_retry(
                client, model, instructions, left_part, max_retries, depth + 1
            )
            text_b, usage_b, _ = generate_with_retry(
                client, model, instructions, right_part, max_retries, depth + 1
            )

            if text_a is None or text_b is None:
                return "Error", dict(), time.perf_counter() - t0 - time_to_first_token

            return (text_a + '|next-item|' + text_b,
                    _sum_usage(usage_a, usage_b),
                    time.perf_counter() - t0 - time_to_first_token
                    )

        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print(f"{indent}[!] 503 Unavailable (Attempt {attempt}/{max_retries}). Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"{indent}Exception Type: {type(e).__name__}")
                print(f"{indent}[X] Critical Error: {e}")
                raise e

    print(f"{indent}[X] Failed after max retries.")
    return "Error", None, time.perf_counter() - t0 - time_to_first_token


def write_results_to_file(input: list[str],
                          computed_output: list[str],
                          expected_output: list[str],
                          info: dict[str, int],  # time is in ms
                          output_file: str,
                          config: RunConfig | None = None) -> None:
    """
    Write the results out after comparing the computed and expected MathML outputs.
    If show_normalized = True, computed_output and expected_output should both be MathML (=> input is braille)
    """
    usage_info = str(info)[1:-1].replace("'", "").replace(": ", "=")
    print(f"Generated {len(computed_output)} outputs. Stats: {usage_info}ms")
    is_mathml_output = expected_output[0].startswith('<math')
    if not isinstance(computed_output, list):
        print(f"Error: Computed output is a {type(computed_output)},\
              {len(computed_output) if isinstance(computed_output, list) else 0} items")
        return
    if is_mathml_output and not computed_output[0].startswith('<math'):
        print("Computed output does not appear to be MathML--first 5 lines:\n", computed_output[:5])
        return
    if not is_mathml_output and not re.match('[\u2800-\u28ff]', computed_output[0][0]):
        print("Computed output does not appear to be MathML--last 5 lines:\n", computed_output[len(computed_output)-5:])
        return

    # initial MathCAT
    setMathCATPreferences({})

    with open(output_file, "w", encoding="utf-8") as f:
        # Write variable values from main() at the start
        if config:
            f.write("# Example files (used in generateExamples):\n")
            f.write(f"# example_braille_file = {config.example_braille_file}\n")
            f.write(f"# example_mathml_file = {config.example_mathml_file}\n")
            f.write("#\n")
            f.write("# Input files:\n")
            f.write(f"# input_braille_file = {config.input_braille_file}\n")
            f.write(f"# input_mathml_file = {config.input_mathml_file}\n")
            f.write("#\n")
            f.write("# Configuration variables:\n")
            f.write(f"# braille_code = {config.braille_code}\n")
            f.write(f"# gen_braille = {config.gen_braille}\n")
            f.write(f"# model = {config.model}\n")
            f.write(f"# apiKeyName = {config.apiKeyName}\n")
            f.write(f"# batch_size = {config.batch_size}\n")
            f.write(f"# n_examples = {config.n_examples}\n")
            if config.chunk:
                f.write(f"# chunk = slice({config.chunk.start}, {config.chunk.stop})\n")
            f.write("#\n")
            if config.instructions:
                # Write instructions with proper line breaks and # prefix
                f.write("# instructions = \n")
                for line in config.instructions.split('\n'):
                    f.write(f"# {line}\n")
            f.write("#\n")

        match_count = 0
        f.write(f"# {len(computed_output)} items. "
                f"Usage info: {usage_info}ms, "
                f"TPS={(1000 * info['time']/info['candidates'])}.2f\n#\n")
        if is_mathml_output:
            f.write("\n# NOT Normalized MathML\n")
        f.write("# Match | Test Input | Expected | Computed\n")
        for tests, computed, expected in zip(input, computed_output, expected_output):
            try:
                if is_mathml_output:
                    checked = areCanonicallyEqual(expected, computed)
                else:
                    checked = CanonicalResults(expected.strip() == computed.strip(), "", "")
                if checked.isEqual:
                    match_count += 1
            except Exception:
                checked = CanonicalResults(False, "", "")
            match = "✓" if checked.isEqual else "✗"
            f.write(f"{match} | {tests} | {expected} | {computed}\n")
        if is_mathml_output:
            f.write("\n#===========\n")
            f.write("\n# Normalized MathML\n")
            f.write("Match | Test Input | Expected | Computed\n")
            for tests, computed, expected in zip(input, computed_output, expected_output):
                try:
                    checked = areCanonicallyEqual(expected, computed)
                except Exception as e:
                    print(f"areCanonicallyEqual error message:\n{e}", file=sys.stderr)
                    checked = CanonicalResults(False, expected, '<--bad MathML-->' + computed)

                match = "✓" if checked.isEqual else "✗"
                f.write(f"{match} | {tests} | {checked.canonicalOriginal} | {checked.canonicalComputed}\n")

        f.write(f"# Matches: {match_count} out of {len(input)}: {(match_count/len(input)*100):.0f}%.")
        print(f"Matches: {match_count} out of {len(input)}: {(match_count/len(input)*100):.0f}%. "
              f"Results written to {output_file}. ")


def readMatchingFiles(braille_path: str, mathml_path: str) -> tuple[list[str], list[str]]:
    """
    Reads lines from two files or directories and returns tuple of (braille_lines, mathml_lines).
    If directories are provided, matches files by base name (without extension) and combines all pairs.
    """
    braille_lines = []
    mathml_lines = []

    # Check if paths are directories
    if os.path.isdir(braille_path) and os.path.isdir(mathml_path):
        # Iterate through braille files and read matching pairs directly
        for filename in os.listdir(braille_path):
            if filename.endswith('.brls'):
                base_name = os.path.splitext(filename)[0]
                braille_file = os.path.join(braille_path, filename)
                mathml_file = os.path.join(mathml_path, base_name + '.mmls')

                # If matching mathml file exists, read both files
                if os.path.exists(mathml_file):
                    with open(braille_file, "r", encoding="utf-8") as f:
                        braille_lines.extend(f.read().splitlines())
                    with open(mathml_file, "r", encoding="utf-8") as f:
                        mathml_lines.extend(f.read().splitlines())
    else:
        # Handle as files (original behavior)
        with open(braille_path, "r", encoding="utf-8") as f:
            braille_lines = f.read().splitlines()
        with open(mathml_path, "r", encoding="utf-8") as f:
            mathml_lines = f.read().splitlines()

    return braille_lines, mathml_lines


def generateExamples(braille_path, mathml_path) -> str:
    """
    Zips lines from two files or directories and returns a string of the form "braille | mathml\n".
    If directories are provided, matches files by base name (without extension) and combines all pairs.
    """
    try:
        braille_lines, mathml_lines = readMatchingFiles(braille_path, mathml_path)

        # Zip and format
        result = ""
        for brl, mml in zip(braille_lines, mathml_lines):
            # .strip() removes the trailing newline character for cleaner output
            quoted_mml = mml.strip().replace('"', '\\"')
            result += f'"{brl.strip()} | {quoted_mml}\\n"\n'

        return result

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e.filename}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return ""


def getInstructionsProlog(gen_braille: bool, braille_code: str) -> str:
    """
    Returns the instructions prolog based on whether generating braille or MathML.

    Args:
        gen_braille: If True, generates braille from MathML. If False, generates MathML from braille.
        braille_code: The braille code to use (e.g., "Nemeth", "UEB")

    Returns:
        The instructions prolog string
    """
    if gen_braille:
        return (
            f"You are an expert Braille translator specializing in {braille_code} braille. "
            "The user will provide a python list of strings, "
            "where each string is composed of represents math encoded in MathML "
            "(e.g., <math><mi>f</mi><mrow><mo>(</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo>)</mo></mrow></math>). "
            f"Your task is to translate each MathML expression into valid {braille_code} braille "
            "using only Unicode braille characters. "
            "For each MathML input, output ONLY the raw Unicode braille characters. (e.g., '⠋⠷⠹⠂⠌⠆⠼⠾')"
            "It is important to pay attention to generating Unicode braille spaces when needed in the braille. "
            "It is also important to pay attention when to generate "
            f"{'the number sign indicator ⠼ and the English letter indicator' if braille_code == 'Nemeth' else
               'grade 1 indicators ⠰, grade 1 word indicators ⠰⠰, and grade 1 passage indicators ⠰⠰⠰ when appropriate. '
               'These are very common at the start of the translation.'}"
            "Do not include markdown formatting, explanations, or any other text."
            "Add '|next-item|' between each braille output. "
            "Below are some examples of braille/MathML pairs separated by '|' that should be considered the ground truth:\n"
        )
    else:
        return (
            f"You are an expert Braille translator specializing in {braille_code} braille. "
            "The user will provide a python list of strings, "
            f"where each string is composed of Unicode {braille_code} Braille characters.  "
            "Here are two examples: '⠠⠑⠀⠨⠅⠀⠍⠉⠘⠆' and  '⠎⠊⠝⠀⠷⠨⠹⠾'. "
            "Your task is to translate each exact Braille sequence of characters into valid MathML code. "
            "For each braille input, output ONLY the raw MathML string starting with <math> and ending with </math>. "
            "Every element in the MathML must be properly closed and nested. "
            "Do not include markdown formatting, explanations, or any other text. "
            "Do not include any newlines or carriage returns. "
            "Add '|next-item|' between each MathML output. "
            "Below are some examples of braille/MathML pairs separated by '|' "
            "that should be considered the ground truth:\n"
        )


def run_conversion(
    gen_braille: bool,
    braille_code: str,
    chunk_size: int | None = None
) -> None:
    """
    Run the conversion process to generate braille or MathML.

    Args:
        gen_braille: If True, generates braille from MathML. If False, generates MathML from braille.
        braille_code: The braille code to use (e.g., "Nemeth", "UEB")
        chunk_size: Number of items to process (defaults to all items if None)
    """
    # Get instructions prolog based on output type
    instructions_prolog = getInstructionsProlog(gen_braille, braille_code)

    # File paths for examples
    example_braille_file = f"RustTestData/{braille_code}.brls"
    example_mathml_file = f"RustTestData/{braille_code}.mmls"
    # example_mathml_file = f"RustTestData/{braille_code}-cnclz.mmls"
    examples = generateExamples(example_braille_file, example_mathml_file)

    additional_examples = generateExamples(
        f"example_data/{braille_code.lower()}.brls",
        f"example_data/mathml.mmls"
    )
    truncated_additional_examples = additional_examples.splitlines(keepends=True)
    additional_examples = "".join(truncated_additional_examples[:1000])
    examples += "\n" + additional_examples

    test_mathml_dir = "test_data/MathML"
    test_braille_dir = f"test_data/{braille_code}"

    # File paths for input - gather lines from directories
    braille, mathml = readMatchingFiles(test_braille_dir, test_mathml_dir)
    if len(braille) != len(mathml):
        print("Error: Number of test inputs does not match number of expected outputs.")
        sys.exit(1)

    # Use chunk_size parameter, default to len(mathml) if not provided
    chunk_size = chunk_size if chunk_size is not None else len(mathml)
    chunk = slice(0, chunk_size)
    braille = braille[chunk]
    mathml = mathml[chunk]
    batch_size = 80
    # model = "gemini-3-flash-preview"
    # model = "gemini-2.5-flash"
    # model = "gemini-2.5-flash-lite"
    model = "gemini-2.5-pro"
    # model = "gemini-3-pro-preview"
    apiKeyName = "GEMINI_API_KEY"
    # apiKeyName = "GEMINI_PAID_API_KEY"
    print(f"Using API key: {apiKeyName}")
    n_examples = examples.count('\n')

    # GENERATE either braille or MathML
    instructions = instructions_prolog + examples
    input, expected = (mathml, braille) if gen_braille else (braille, mathml)

    print(f"Generating {'braille' if gen_braille else 'MathML'} with {n_examples} examples, "
          f"{len(input)} tests with {model} for {braille_code}.")
    try:
        computed, total_tokens, total_generation_time = convert_input_with_model(
            instructions, input, model, apiKeyName, batch_size
        )
        if computed is None:
            computed = []
        total_tokens['time'] = round(1000 * total_generation_time)  # ms -- needs to be an int
        config = RunConfig(
            braille_code=braille_code,
            gen_braille=gen_braille,
            model=model,
            apiKeyName=apiKeyName,
            batch_size=batch_size,
            n_examples=n_examples,
            chunk=chunk,
            instructions=instructions,
            example_braille_file=example_braille_file,
            example_mathml_file=example_mathml_file,
            input_braille_file=test_braille_dir,
            input_mathml_file=test_mathml_dir
        )
        output_filename = (
            f"{'to-' if gen_braille else 'from-'}{braille_code}-{model}-"
            f"{n_examples}exs-{len(input)}tests.txt"
        )
        write_results_to_file(input, computed, expected, total_tokens,
                              output_filename, config=config)
    except Exception as e:
        print(f"Conversion error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate braille or MathML using Gemini API',
        epilog='''
Examples:
  # Generate MathML from Nemeth braille, process first 200 items:
  python use_gemini.py mathml Nemeth 200

  # Generate braille from MathML using UEB, process all items:
  python use_gemini.py braille UEB

  # Generate MathML from Nemeth braille, process all items:
  python use_gemini.py mathml Nemeth

Note: Requires GEMINI_API_KEY or GEMINI_PAID_API_KEY environment variable.
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('output_type', type=str,
                        help=('Output type: "braille" or "mathml" (case insensitive). '
                              'If "mathml", generates MathML from braille. '
                              'If "braille", generates braille from MathML.'))
    parser.add_argument('braille_code', type=str,
                        help='Braille code to use. Options: Nemeth, UEB, etc.')
    parser.add_argument('chunk_size', type=int, nargs='?', default=None,
                        help=('Number of items to process (defaults to all items if not provided). '
                              'Useful for testing with a subset of data.'))

    args = parser.parse_args()

    # Parse output_type to determine gen_braille
    output_type_lower = args.output_type.lower()
    if output_type_lower not in ('braille', 'mathml'):
        parser.error(
            f'output_type must be "braille" or "mathml", got "{args.output_type}". '
            f'Example: python use_gemini.py mathml Nemeth 200'
        )
    gen_braille = output_type_lower == 'braille'

    # Call the refactored function with parsed arguments
    run_conversion(gen_braille, args.braille_code, args.chunk_size)


if __name__ == "__main__":
    main()
