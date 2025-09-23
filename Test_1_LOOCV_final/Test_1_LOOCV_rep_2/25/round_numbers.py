import re
import sys

# Function to process a single file
def process_file(filepath):
    try:
        # Read the file
        with open(filepath, "r") as file:
            lines = file.readlines()

        # Regular expression to match probabilistic facts like "0.1234567::uX"
        prob_fact_pattern = re.compile(r"(\d+\.\d+)(::u\d+\.)")

        # Process each line
        modified_lines = []
        for line in lines:
            # Check if the line contains a probabilistic fact
            match = prob_fact_pattern.search(line)
            if match:
                # Extract the number part
                number = float(match.group(1))
                # Round to 7 decimal places
                rounded_number = f"{number:.7f}"
                # Replace the original number with the rounded one
                modified_line = line.replace(match.group(1), rounded_number)
                modified_lines.append(modified_line)
            else:
                # Keep non-probabilistic lines unchanged
                modified_lines.append(line)

        # Write the modified content back to the file
        with open(filepath, "w") as file:
            file.writelines(modified_lines)
        
        print(f"Successfully processed {filepath}")

    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")

# Check if a filename was provided as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python round_probabilities_single.py <filename>")
    sys.exit(1)

# Get the filename from command-line arguments
filename = sys.argv[1]

# Process the file
process_file(filename)
