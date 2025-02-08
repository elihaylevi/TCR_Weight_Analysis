############################################################
# STEP 2: Calculate TCR Weight and Clonality Scores
# - Extract TCR clonality from validation data
# - Compute statistical properties of TCRs
# - Filter TCRs appearing in >5% of samples
# - Compute TCR weighting metrics (consistency, variance)
############################################################

import os
import pandas as pd
import concurrent.futures
import ast
import numpy as np
from collections import defaultdict

# Define directories
DATA_DIR = '/dsi/scratch/home/dsi/elihay/dean/downsampled_files'
OUTPUT_DIR = '/dsi/scratch/home/dsi/elihay/dean/outputs'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of parallel processes
NUM_PROCESSES = 64

############################################################
# STEP 1: Extract TCR Clonality from Validation Data
############################################################

# Function to process each downsampled file
def process_downsampled_file(file_path):
    try:
        # Read 'amino_acid' and 'templates' columns
        df = pd.read_csv(file_path, delimiter="\t", usecols=['amino_acid', 'templates'])
        # Convert to dictionary with amino_acid as keys and templates as values
        tcr_counts = df.set_index('amino_acid')['templates'].to_dict()
        return tcr_counts
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Initialize a defaultdict to store counts and a list to track errors
tcr_sample_counts = defaultdict(list)
error_files = []

# Get the list of downsampled files with '.tsv' extension
downsampled_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.tsv')]

# Process files in parallel using ProcessPoolExecutor
with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
    for file_path, file_counts in zip(downsampled_files, executor.map(process_downsampled_file, downsampled_files)):
        if file_counts is not None:
            for tcr, count in file_counts.items():
                tcr_sample_counts[tcr].append(count)
        else:
            error_files.append(file_path)

# Convert clonality data to DataFrame
tcr_counts_df = pd.DataFrame([(k, len(v), v) for k, v in tcr_sample_counts.items()], 
                             columns=['amino_acid', 'sample_appearance_count', 'sample_counts'])

# Save TCR clonality data
clonality_output = os.path.join(OUTPUT_DIR, 'tcr_counts_per_sample_train.csv')
tcr_counts_df.to_csv(clonality_output, index=False)
print(f"TCR clonality data saved to {clonality_output}")

# Log files with errors
if error_files:
    print("Files with errors:")
    for file in error_files:
        print(file)

############################################################
# STEP 2: Compute Statistical Properties of TCRs
############################################################

# Load the extracted TCR clonality data
df = pd.read_csv(clonality_output)

# Function to compute statistical properties
def calculate_statistics(dataframe):
    return {
        'Median': dataframe.median(),
        'Average': dataframe.mean(),
        'Standard Deviation': dataframe.std(),
        'Variance': dataframe.var(),
        'Skewness': dataframe.skew(),
        'Kurtosis': dataframe.kurt(),
        '90th Percentile': dataframe.quantile(0.90),
        '99th Percentile': dataframe.quantile(0.99),
        '99.9th Percentile': dataframe.quantile(0.999)
    }

# Compute statistics for all TCRs
all_data_stats = calculate_statistics(df['sample_appearance_count'])

# Compute statistics excluding TCRs that appear in only 1 sample
df_excluding_ones = df[df['sample_appearance_count'] > 1]
excluding_ones_stats = calculate_statistics(df_excluding_ones['sample_appearance_count'])

# Compute statistics excluding TCRs that appear in 1-10 samples
df_excluding_1_to_10 = df[df['sample_appearance_count'] > 10]
excluding_1_to_10_stats = calculate_statistics(df_excluding_1_to_10['sample_appearance_count'])

# Compute sample percentage for low-frequency TCRs
percentage_of_ones = (df['sample_appearance_count'] == 1).mean() * 100
percentage_of_one_to_10 = (df['sample_appearance_count'].between(1, 10)).mean() * 100

# Create a DataFrame for statistics
statistics_df = pd.DataFrame({
    'All Data': all_data_stats,
    'Excluding 1': excluding_ones_stats,
    'Excluding 1 to 10': excluding_1_to_10_stats,
    'Percentage of 1': percentage_of_ones,
    'Percentage of 1 to 10': percentage_of_one_to_10
})

# Save statistics data
statistics_output = os.path.join(OUTPUT_DIR, 'tcr_statistics_train.csv')
statistics_df.to_csv(statistics_output)
print(f"TCR statistics saved to {statistics_output}")

############################################################
# STEP 3: Filter TCRs Appearing in More than 5% of Samples
############################################################

# Define filtering threshold (5% of total samples)
threshold = int(len(df) * 0.05)

# Select TCRs appearing in >5% of samples
df_filtered = df[df['sample_appearance_count'] > threshold]

# Save filtered TCRs
filtered_output = os.path.join(OUTPUT_DIR, 'tcr_counts_per_sample_train_over24.csv')
df_filtered.to_csv(filtered_output, index=False)
print(f"Filtered TCRs saved to {filtered_output}")

############################################################
# STEP 4: Compute TCR Weighting Metrics (Consistency & Variance)
############################################################

# Define file paths
input_file = filtered_output
output_file = os.path.join(OUTPUT_DIR, 'tcr_counts_and_cloness_with_variance24.csv')

# Load filtered TCR data
df = pd.read_csv(input_file)

# Function to compute clonality scores and variance
def calculate_scores_and_variance(batch):
    results = []
    for row in batch:
        # Safely parse the list of sample counts
        counts = ast.literal_eval(row['sample_counts'])
        sample_counts = len(counts)
        average_count = sum(counts) / sample_counts if sample_counts > 0 else 0
        
        # Adjusted consistency score (threshold = 10)
        consistency_score = sum(min(count, 10) for count in counts)
        consistency_area = consistency_score / (10 * sample_counts) if sample_counts > 0 else 0

        # Compute variance
        variance_score = np.var(counts) if sample_counts > 0 else 0

        # Append computed metrics
        results.append((consistency_score, consistency_area, average_count, variance_score))
    return results

# Apply multiprocessing for efficiency
num_processes = 60
batch_size = len(df) // num_processes
batches = [df.iloc[i:i + batch_size].to_dict('records') for i in range(0, len(df), batch_size)]

with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    results = list(executor.map(calculate_scores_and_variance, batches))

# Flatten results and add to DataFrame
flat_results = [item for sublist in results for item in sublist]
df[['consistency_score', 'consistency_area', 'average_count', 'variance_score']] = pd.DataFrame(flat_results, index=df.index)

# Save final TCR weighting data
df.to_csv(output_file, index=False)
print(f"TCR weighting data saved to {output_file}")

print("Step 2 (Calculate TCR Weight) complete.")
