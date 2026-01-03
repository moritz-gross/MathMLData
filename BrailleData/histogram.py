import pandas as pd
import glob
import os
import statistics
from typing import List

def generate_line_histogram(
    directory: str, 
    file_pattern: str = "*.txt", 
    output_name: str = "histogram_output.xlsx"
) -> None:
    """
    Reads multiple UTF-8 files, prints summary statistics, and exports a 
    histogram to an Excel file.
    """
    all_lengths: List[int] = []
    
    search_path = os.path.join(directory, file_pattern)
    files = glob.glob(search_path)
    
    if not files:
        print(f"No files found matching pattern: {search_path}")
        return

    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                # Count characters per line, excluding trailing whitespace
                lengths = [len(line.strip()) for line in f if len(line.strip()) > 2]
                all_lengths.extend(lengths)
        except Exception as e:
            print(f"Could not read file {file}: {e}")

    if not all_lengths:
        print("No data found in the files.")
        return

    # --- Calculation of Statistics ---
    total_lines = len(all_lengths)
    total_chars = sum(all_lengths)
    mean_len = statistics.mean(all_lengths)
    median_len = statistics.median(all_lengths)

    print("-" * 30)
    print(f"Summary Statistics for '{directory}':")
    print(f"Total Lines:      {total_lines}")
    print(f"Total Characters: {total_chars}")
    print(f"Mean Length:      {mean_len:.2f}")
    print(f"Median Length:    {median_len}")
    print("-" * 30)

    # --- Excel Generation ---
    df = pd.DataFrame(all_lengths, columns=['LineLength'])
    hist_data = df['LineLength'].value_counts().sort_index().reset_index()
    hist_data.columns = ['Character Count', 'Frequency']

    writer = pd.ExcelWriter(output_name, engine='xlsxwriter')
    hist_data.to_excel(writer, sheet_name='HistogramData', index=False)

    workbook  = writer.book
    worksheet = writer.sheets['HistogramData']
    chart = workbook.add_chart({'type': 'column'})
    max_row = len(hist_data)

    chart.add_series({
        'name':       'Line Length Frequency',
        'categories': ['HistogramData', 1, 0, max_row, 0],
        'values':     ['HistogramData', 1, 1, max_row, 1],
        'fill':       {'color': '#4F81BD'}
    })

    chart.set_title({'name': 'Distribution of Line Lengths'})
    chart.set_x_axis({'name': 'Number of Characters (UTF-8)'})
    chart.set_y_axis({'name': 'Frequency'})
    chart.set_legend({'position': 'none'})

    worksheet.insert_chart('D2', chart)
    writer.close()
    
    print(f"Successfully generated '{output_name}'.")

# --- Example Usage ---
# --- Example Usage ---
if __name__ == "__main__":
    generate_line_histogram(directory="Braille/LaTeX/highschool", file_pattern="*.brls")