import os
import statistics
from pathlib import Path

def count_in_file(file_path):
    """Count ':-' and 'latent_collision :-' in a single file."""
    if not file_path.exists():
        print(f"  Warning: {file_path} not found → skipped")
        return None, None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    rule_count = content.count(':-')
    latent_count = content.count('latent_collision :-')
    
    return rule_count, latent_count

def process_directory(dir_path):
    """Process all cBN_1.pl to cBN_768.pl in the given directory."""
    rule_counts = []
    latent_counts = []
    
    base = Path(dir_path)
    print(f"\nProcessing: {base}")
    
    for i in range(1, 769):
        file_path = base / f"cBN_{i}.pl"
        counts = count_in_file(file_path)
        if counts[0] is not None:
            rule_counts.append(counts[0])
            latent_counts.append(counts[1])
    
    if not rule_counts:
        print("  No valid files found.")
        return
    
    # Statistics
    avg_rules   = statistics.mean(rule_counts)
    std_rules   = statistics.stdev(rule_counts) if len(rule_counts) > 1 else 0.0
    avg_latent  = statistics.mean(latent_counts)
    std_latent  = statistics.stdev(latent_counts) if len(latent_counts) > 1 else 0.0
    
    n_files = len(rule_counts)
    
    print(f"  Files processed: {n_files}/768")
    print(f"  Average number of logic rules (':-'):      {avg_rules:8.3f}  ± {std_rules:6.3f}")
    print(f"  Average 'latent_collision :-' occurrences: {avg_latent:8.3f}  ± {std_latent:6.3f}")
    print("-" * 60)

# The three directories (adjust paths if needed)
directories = [
    "../rep_1/01/cBNs",
    "../rep_1/50/cBNs",
    "../rep_1/100/cBNs"
]

print("Analyzing number of rules per training percentage\n")
for d in directories:
    process_directory(d)
