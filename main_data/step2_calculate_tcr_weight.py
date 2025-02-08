######################################################################
#  STEP 2: Extracting TCR Clonality & Computing TCR Weight
# - Aggregates TCR occurrences across samples
# - Calculates key clonality metrics (consistency, explosion, variance)
# - Filters TCRs appearing in more than 5% of samples
######################################################################

######################################################
# Extract TCR Clonality Across Samples
# - Reads downsampled TCR files
# - Counts occurrences of each TCR in different samples
# - Saves aggregated clonality data to CSV
######################################################

import os
import pandas as pd
from collections import defaultdict
import concurrent.futures

# Define input and output directories
train_path = '/dsi/scratch/home/dsi/elihay/downsampled_files/train/'
output_dir = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the number of parallel processes
num_processes = 64

# Function to process each TCR file and extract clonality data
def process_downsampled_file(file_path):
    try:
        df = pd.read_csv(file_path, delimiter="\t", usecols=['amino_acid', 'templates'])
        tcr_counts = df.set_index('amino_acid')['templates'].to_dict()
        return tcr_counts
    except Exception as e:
        return None

# Dictionary to store TCR counts across samples
tcr_sample_counts = defaultdict(list)
error_files = []

# List all TCR files in the train directory
downsampled_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.tsv')]

# Process TCR files in parallel using multiprocessing
with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    for file_counts in executor.map(process_downsampled_file, downsampled_files):
        if file_counts is not None:
            for tcr, count in file_counts.items():
                tcr_sample_counts[tcr].append(count)
        else:
            error_files.append(file_name)

# Convert TCR sample counts to DataFrame and save to CSV
tcr_counts_df = pd.DataFrame([(k, len(v), v) for k, v in tcr_sample_counts.items()], 
                             columns=['amino_acid', 'sample_appearance_count', 'sample_counts'])

output_file = os.path.join(output_dir, 'tcr_counts_per_sample_train.csv')
tcr_counts_df.to_csv(output_file, index=False)
print(f"TCR clonality data saved to {output_file}")

######################################################
# Compute Clonality Statistics
# - Calculates median, mean, variance, skewness, and percentiles
# - Analyzes distributions with and without rare TCRs
######################################################

import pandas as pd

# Load TCR clonality data
file_path = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/tcr_counts_per_sample_train.csv'
df = pd.read_csv(file_path)

# Function to compute statistical metrics
def calculate_statistics(dataframe):
    stats = {
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
    return stats

# Compute statistics for all TCRs
all_data_stats = calculate_statistics(df['sample_appearance_count'])

# Compute statistics excluding single-instance TCRs
df_excluding_ones = df[df['sample_appearance_count'] > 1]
excluding_ones_stats = calculate_statistics(df_excluding_ones['sample_appearance_count'])

# Compute statistics excluding TCRs appearing 1-10 times
df_excluding_1_to_10 = df[df['sample_appearance_count'] > 10]
excluding_1_to_10_stats = calculate_statistics(df_excluding_1_to_10['sample_appearance_count'])

# Calculate proportions of rare TCRs
percentage_of_ones = (df['sample_appearance_count'] == 1).mean() * 100
percentage_of_one_to_10 = (df['sample_appearance_count'].between(1, 10)).mean() * 100

# Combine all statistics into a DataFrame and save to CSV
statistics = {
    'All Data': all_data_stats,
    'Excluding 1': excluding_ones_stats,
    'Excluding 1 to 10': excluding_1_to_10_stats,
    'Percentage of 1': percentage_of_ones,
    'Percentage of 1 to 10': percentage_of_one_to_10
}

statistics_df = pd.DataFrame(statistics)
output_file = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/tcr_statistics_train.csv'
statistics_df.to_csv(output_file)
print(f"Clonality statistics saved to {output_file}")

######################################################
# Compute TCR Weight: Consistency & Explosion Scores
# - Calculates weighted measures of clonality
# - Computes scores for consistency and variance
# - Stores results in a new CSV file
######################################################

import pandas as pd
import concurrent.futures
import ast
import numpy as np

def calculate_batch_scores(batch):
    results = []
    for row in batch:
        # Convert stored string representation of list back to list
        counts = ast.literal_eval(row['sample_counts'])
        sample_counts = len(counts)
        average_count = sum(counts) / sample_counts if sample_counts > 0 else 0
        
        # Calculate TCR weight metrics
        consistency_score = sum(min(count, 12) for count in counts)
        consistency_score2 = sum(min(count, 6) for count in counts)
        explosion_score = sum(min(count, 433) for count in counts)
        
        # Normalize scores based on sample counts
        consistency_area = consistency_score / (12 * sample_counts) if sample_counts > 0 else 0
        explosion_area = explosion_score / (433 * sample_counts) if sample_counts > 0 else 0
        
        results.append((consistency_score, consistency_score2, explosion_score, consistency_area, explosion_area, average_count))
    return results

# Load clonality data
file_path = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/tcr_counts_per_sample_train.csv'
df = pd.read_csv(file_path)

# Parallel processing to compute TCR weight scores
num_processes = 60
batch_size = len(df) // num_processes
batches = [df.iloc[i:i + batch_size].to_dict('records') for i in range(0, len(df), batch_size)]

with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    results = list(executor.map(calculate_batch_scores, batches))

# Flatten results and add to DataFrame
flat_results = [item for sublist in results for item in sublist]
df[['consistency_score', 'consistency_score2', 'explosion_score', 'consistency_area', 'explosion_area', 'average_count']] = pd.DataFrame(flat_results, index=df.index)

# Save updated dataset
output_file = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/tcr_counts_and_cloness.csv'
df.to_csv(output_file, index=False)
print(f"TCR weight scores saved to {output_file}")

######################################################
# Filter TCRs Appearing in More Than 5% of Samples
# - Retains high-frequency TCRs
# - Removes rare TCRs that appear in fewer than 5% of cases
######################################################

import pandas as pd 

# Load TCR clonality dataset
file_path = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/tcr_counts_and_cloness.csv'
df = pd.read_csv(file_path)

# Keep only TCRs appearing in at least 5% of samples
min_threshold = 122  # Adjust based on dataset size
df_filtered = df[df['sample_appearance_count'] > min_threshold]

# Save the filtered dataset
output_file = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/tcr_counts_filtered.csv'
df_filtered.to_csv(output_file, index=False)
print(f"Filtered TCR dataset saved to {output_file}")
