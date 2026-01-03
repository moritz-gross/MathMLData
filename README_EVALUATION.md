# Nemeth Braille to MathML Translation Evaluation

This script evaluates the translation ability of OpenAI's o3-mini reasoning model from Nemeth Braille to MathML using few-shot prompting.

## Overview

The script:
- Loads paired Nemeth Braille and MathML expressions from the highschool dataset
- Filters pairs by MathML length (20-200 characters)
- Randomly samples 100 test pairs
- For each test pair, uses 10 randomly selected examples as few-shot demonstrations
- Calls o3-mini to translate Nemeth Braille → MathML
- Exports results to a comprehensive CSV file with metadata

## Requirements

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key with access to o3-mini (requires API tier 3-5)

## Installation

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:
```bash
uv sync
```

3. Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

Run the script with uv:
```bash
uv run python nemeth_to_mathml_evaluation.py
```

Or activate the virtual environment and run directly:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python nemeth_to_mathml_evaluation.py
```

The script will:
1. Load all highschool Nemeth Braille files (`*-no-dups.brls`)
2. Filter pairs with MathML length between 20-200 characters
3. Sample 100 random pairs for testing
4. Process each pair with o3-mini using 10 few-shot examples
5. Save results to `nemeth_mathml_evaluation_results.csv`

## Configuration

You can modify the following constants in the script:

```python
# Filter parameters
MIN_MATHML_LENGTH = 20          # Minimum MathML character length
MAX_MATHML_LENGTH = 200         # Maximum MathML character length

# Sampling parameters
NUM_FEW_SHOT_EXAMPLES = 10      # Number of examples in each prompt
NUM_TEST_PAIRS = 100            # Number of pairs to evaluate

# Model configuration
MODEL_NAME = "o3-mini"          # OpenAI model to use
REASONING_EFFORT = "high"       # Reasoning effort: "low", "medium", or "high"
MAX_TOKENS = 2000               # Maximum tokens in response

# Random seed
RANDOM_SEED = 42                # For reproducibility
```

## Output

### CSV File Structure

The output CSV (`nemeth_mathml_evaluation_results.csv`) contains the following columns:

| Column | Description |
|--------|-------------|
| `id` | Sequential ID (1 to 100) |
| `source_file` | Source Braille filename |
| `line_number` | Line number in source file |
| `nemeth_braille` | Input Nemeth Braille expression |
| `nemeth_length` | Character count of Braille input |
| `ground_truth_mathml` | True MathML from dataset |
| `ground_truth_length` | Character count of ground truth |
| `predicted_mathml` | Model's predicted MathML output |
| `model_name` | Model used (e.g., "o3-mini") |
| `timestamp` | ISO 8601 timestamp of API call |
| `prompt_tokens` | Number of tokens in prompt |
| `completion_tokens` | Number of tokens in completion |
| `total_tokens` | Total tokens used |
| `reasoning_tokens` | Internal reasoning tokens (if available) |
| `reasoning_effort` | Reasoning effort level ("low", "medium", or "high") |
| `response_time_seconds` | API call duration |
| `few_shot_example_indices` | JSON list of example locations used |
| `error` | Error message (null if successful) |

### Progress Tracking

The script uses **tqdm** for visual progress bars:
- Shows real-time progress bar for file loading
- Displays evaluation progress with current file/line info
- Saves checkpoints every 10 pairs
- Displays summary statistics at the end

### Example Output

```
================================================================================
Nemeth Braille to MathML Translation Evaluation
================================================================================
Model: o3-mini
Reasoning effort: high
Filter: MathML length 20-200 characters
Few-shot examples: 10
Test pairs: 100
Random seed: 42
================================================================================

Step 1: Initializing OpenAI client...
  ✓ OpenAI client initialized

Step 2: Loading highschool data...
Found 20 Braille files in highschool directory
Loading files: 100%|██████████| 20/20 [00:02<00:00, 8.45file/s]
  Trigonometry-no-dups.brls: 143 pairs (after filtering)
  ...
  ✓ Loaded 1847 filtered pairs

Step 3: Sampling 100 test pairs...
  ✓ Sampled 100 test pairs

Step 4: Running evaluation...
Evaluating: 23%|██▎       | 23/100 [01:12<03:51, 3.01s/pair] Trigonometry-no-dups.brls:42
  → Checkpoint: saving intermediate results...
...

Step 5: Writing final results...
Results written to: nemeth_mathml_evaluation_results.csv
Total rows: 100

================================================================================
SUMMARY
================================================================================
Total pairs processed: 100
Successful: 98
Failed: 2

Token Usage:
  Average prompt tokens: 3247
  Average completion tokens: 187
  Total tokens used: 336,532

Performance:
  Average response time: 2.84s

Output file: nemeth_mathml_evaluation_results.csv
================================================================================
```

## Data Sources

The script uses:
- **Braille files**: `BrailleData/Braille/Nemeth/highschool/*-no-dups.brls`
- **MathML files**: `SimpleSpeakData/highschool/*-no-dups.mmls`

Pairs are matched by filename and line number (line N in `.brls` corresponds to line N in `.mmls`).

## Notes

- **No Evaluation**: Since MathML output may not be unique, the script does not perform automated evaluation. Manual analysis of the CSV is required.
- **Reproducibility**: Set `RANDOM_SEED` for consistent sampling across runs
- **Reasoning Effort**: o3-mini supports three levels - "low" (fastest), "medium" (balanced), "high" (most thorough). Script uses "high" by default.
- **Costs**: With ~3400 tokens per request × 100 pairs = ~340k tokens total. Check OpenAI pricing for o3-mini at current rates.
- **Rate Limiting**: The script includes a 0.5-second delay between API calls to avoid rate limits

## Troubleshooting

### "OPENAI_API_KEY environment variable not set"
Set your API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### "No -no-dups.brls files found"
Ensure you're running the script from the MathMLData project root directory.

### API Errors
- Check your API key has access to o3-mini (requires API tier 3-5)
- Verify you have sufficient API credits
- Check OpenAI service status
- If you don't have access to o3-mini, edit the script to use "o1", "o1-mini", or "gpt-4o"

### Import Error: openai or tqdm
Install dependencies using uv:
```bash
uv sync
```

## License

This evaluation script is part of the MathMLData project (MIT License).
