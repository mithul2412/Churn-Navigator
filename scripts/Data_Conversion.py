import csv
from pathlib import Path

# Configure input and output folders
INPUT_FOLDER = Path("../data/asc_format")  # Folder containing ASC files
OUTPUT_FOLDER = Path("../data/csv_format")  # Folder for output CSV files

# List of files to convert
files_to_convert = [
    'account.asc',
    'card.asc',
    'client.asc',
    'disp.asc',
    'district.asc',
    'loan.asc',
    'order.asc',
    'trans.asc'
]

def convert_asc_to_csv(input_file, output_file):
    """
    Convert an ASC file (semicolon-delimited) to CSV format.
    
    Args:
        input_file (str): Path to the input ASC file
        output_file (str): Path to the output CSV file
    """
    # Read the ASC file with semicolons as delimiters
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read()
    
    # Parse the content row by row, handling quoted fields correctly
    rows = []
    for line in content.strip().split('\n'):
        fields = []
        in_quotes = False
        current_field = ""
        
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
                current_field += char
            elif char == ';' and not in_quotes:
                fields.append(current_field)
                current_field = ""
            else:
                current_field += char
        
        # Add the last field
        fields.append(current_field)
        rows.append(fields)

    # Write to CSV file
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        for row in rows:
            writer.writerow(row)
    
    print(f"Successfully converted {input_file} to {output_file}")

if __name__ == "__main__":
    print("Starting ASC to CSV conversion...")
    success_count = 0
    
    # Create output directory if it doesn't exist
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    for asc_file in files_to_convert:
        # Create full paths for input and output files
        input_path = INPUT_FOLDER / asc_file
        output_path = OUTPUT_FOLDER / asc_file.replace('.asc', '.csv')
        
        convert_asc_to_csv(input_path, output_path)
        success_count += 1
    
    # Print summary
    print(f"\nConversion summary:")
    print(f"Total files to convert: {len(files_to_convert)}")
    print(f"Successfully converted: {success_count}")
    print(f"Failed conversions: {len(files_to_convert) - success_count}")
    print("Conversion process completed.")