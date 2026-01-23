import time
import logging
import re
from typing import Any  # 'Any' is still imported from typing
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
            batch_text, batch_usage, batch_time = generate_with_retry(client, model, instructions, "\n".join(batch), 3)
        except (exceptions.ServiceUnavailable, exceptions.ServerError) as e:
            if first_attempt:
                # reestablish connection and try one more time
                client = genai.Client(api_key=api_key, http_options={"timeout": 2400000})  # isn't allowed in generate_content
                first_attempt = False
                batch_text, batch_usage, batch_time = generate_with_retry(client, model, instructions, "\n".join(batch), 3)
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


def generate_with_retry(
    client: genai.Client,
    model: str,
    instructions: str,
    content: str,
    max_retries: int = 3
) -> tuple[str | None, Any, float]:
    """
    Handles the API call with Stream=True.
    Returns: (generated_text, usage_metadata_object, duration_seconds)
    """

    delay: int = 2

    t0 = time.time()
    for attempt in range(1, max_retries + 1):
        try:

            response_stream = client.models.generate_content_stream(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=instructions,
                    temperature=0.1,
                ),
                contents=content,
            )

            full_text: list[str] = []
            final_usage: Any = None
            finish_reason = None  # Track why it stopped

            for chunk in response_stream:
                if chunk.text:
                    full_text.append(chunk.text)
                if chunk.usage_metadata:
                    final_usage = chunk.usage_metadata

                # Capture the finish reason from the candidates
                if chunk.candidates:
                    finish_reason = chunk.candidates[0].finish_reason

            # CHECK: Did it finish naturally?
            if finish_reason != "STOP":
                raise Exception(f"Incomplete generation: {finish_reason}")

            t1 = time.time()
            duration = t1 - t0

            return "".join(full_text), final_usage, duration

        except (exceptions.ServiceUnavailable, exceptions.ServerError):
            print(f"   [!] 503 Unavailable (Attempt {attempt}/{max_retries}). Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2

        except Exception as e:
            if str(e).startswith('503 UNAVAILABLE'):
                print(f"   [!] 503 Unavailable (Attempt {attempt}/{max_retries}). Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"Exception Type: {type(e).__name__}")
                print(f"Exception Module: {type(e).__module__}")
                print(f"   [X] Critical Error: {e}")
                raise e

    print("   [X] Failed after max retries.")
    return None, None, t1 - t0


def write_results_to_file(input: list[str], computed_output: list[str], expected_output: list[str], show_normalized: bool, info: dict[str, int], output_file: str) -> None:
    """
    Write the results out after comparing the computed and expected MathML outputs.
    If show_normalized = True, computed_output and expected_output should both be MathML (=> input is braille)
    """
    if not isinstance(computed_output, list):
        print(f"Error: Computed output is a {type(computed_output)},\
              {len(computed_output) if isinstance(computed_output, list) else 0} items")
        return
    if  expected_output[0].startswith('<math') and not computed_output[0].startswith('<math'):
        print("Computed output does not appear to be MathML -- first 5 lines:\n", computed_output[:5])
        return
    if re.match('[\u2800-\u28ff]', expected_output[0][0]) and not re.match('[\u2800-\u28ff]', computed_output[0][0]): 
        print("Computed output does not appear to be MathML -- last 5 lines:\n", computed_output[len(computed_output)-5:])
        return

    # initial MathCAT
    setMathCATPreferences({})

    with open(output_file, "w", encoding="utf-8") as f:
        match_count = 0
        f.write(f"Usage info: {info}")
        f.write("\nNOT Normalized MathML\n")
        f.write("Match | Test Input | Expected | Computed\n")
        for tests, computed, expected in zip(input, computed_output, expected_output):
            try:
                checked = areCanonicallyEqual(expected, computed)
                if checked.isEqual:
                    match_count += 1
            except:
                checked = CanonicalResults(False, "", "")
            match = "✓" if checked.isEqual else "✗"
            f.write(f"{match} | {tests} | {expected} | {computed}\n")
        if show_normalized:
            f.write("\nNormalized MathML\n")
            f.write("Match | Test Input | Expected | Computed\n")
            for tests, computed, expected in zip(input, computed_output, expected_output):
                try:
                    checked = areCanonicallyEqual(expected, computed)
                except Exception as e:
                    print(f"areCanonicallyEqual error message:\n{e}", file=sys.stderr)
                    checked = CanonicalResults(False, expected, '<--bad MathML-->' + computed)

                match = "✓" if checked.isEqual else "✗"
                f.write(f"{match} | {tests} | {checked.canonicalOriginal} | {checked.canonicalComputed}\n")

        print(f"Matches: {match_count} out of {len(input)}: {(match_count/len(input)*100):.0f}%. "
              f"Results written to {output_file}. ")


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


def main():
    braille_code = "Nemeth"
    # braille_code = "UEB"
    gen_mathml_instructions_prolog = (
        "You are an expert Braille translator specializing in {braille_code} braille. "
        "The user will provide a python list of strings, "
        "where each string is composed of Unicode {braille_code} Braille characters (e.g., ['⠠⠑⠀⠨⠅⠀⠍⠉⠘⠆', '⠎⠊⠝⠀⠷⠨⠹⠾']). "
        "Your task is to translate each exact Braille sequence of characters into valid MathML code. "
        "For each braille input, output ONLY the raw MathML string starting with <math> and ending with </math>. "
        "Every element in the MathML must be properly closed and nested. "
        "Do not include markdown formatting, explanations, or any other text."
        "Add '|next-item|' between each MathML output. "
        "Below are some examples of braille/MathML pairs separated by '|' that should be considered the ground truth:\n"
    )

    gen_braille_instructions_prolog = (
        f"You are an expert Braille translator specializing in {braille_code} braille. "
        "The user will provide a python list of strings, "
        "where each string is composed of represents math encoded in MathML "
        "(e.g., <math><mi>f</mi><mrow><mo>(</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo>)</mo></mrow></math>). "
        f"Your task is to translate each MathML expression into valid {braille_code} braille using only Unicode braille characters. "
        "For each MathML input, output ONLY the raw Unicode braille characters. (e.g., '⠋⠷⠹⠂⠌⠆⠼⠾')"
        "Do not include markdown formatting, explanations, or any other text."
        "Add '|next-item|' between each braille output. "
        "Below are some examples of braille/MathML pairs separated by '|' that should be considered the ground truth:\n"
    )

    examples = generateExamples(f"RustTestData/{braille_code}.brls", f"RustTestData/{braille_code}.mmls")

    # unicode = read_unicode("xxx")
    with open(f"BrailleData/Braille/{braille_code}/highschool/Statistics-no-dups.brls", "r", encoding="utf-8") as f:
        braille = f.read().splitlines()
    with open("SimpleSpeakData/highschool/Statistics-no-dups.mmls", "r", encoding="utf-8") as f:
        mathml = f.read().splitlines()
    if len(braille) != len(mathml):
        print("Error: Number of test inputs does not match number of expected outputs.")
        sys.exit(1)

    chunk = slice(1326, 1345)  # select a subset for testing
    braille = braille[chunk]
    mathml = mathml[chunk]
    batch_size = 10
    model = "gemini-3-flash-preview"
    model = "gemini-2.5-flash"
    # model = "gemini-2.5-flash-lite"
    # model = "gemini-2.5-pro"
    # model = "gemini-3-pro-preview"
    apiKeyName = "GEMINI_PAID_API_KEY"
    # apiKeyName = "GEMINI_API_KEY"
    n_examples = examples.count('\n')

    gen_braille = True         # GENERATE either braille or MathML
    if gen_braille:
        instructions = gen_braille_instructions_prolog + examples
        input = mathml
        expected = braille
        show_normalized = False

    else:
        instructions = gen_mathml_instructions_prolog + examples
        input = braille
        expected = mathml
        show_normalized = True

    print(f"Generating {'braille' if gen_braille else 'MathML'} with {n_examples} examples, {len(input)} tests with {model} for {braille_code}.")
    try:
        computed, total_tokens, total_generation_time = convert_input_with_model(instructions, input, model, apiKeyName, batch_size)
        if computed is None:
            computed = []
        total_tokens[time] = round(total_generation_time)
        print(f"Generated {len(computed)} in {total_generation_time:1f} secs. Used {total_tokens} tokens")
        write_results_to_file(input, computed, expected, show_normalized, total_tokens, f"{braille_code}-{model}-{n_examples}exs-{len(input)}{braille_code}.txt")
    except Exception as e:
        print(f"Conversion error: {e}")


if __name__ == "__main__":
    main()
