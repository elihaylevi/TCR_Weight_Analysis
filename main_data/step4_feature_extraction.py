##############################################################################
# STEP 4: Creating and Merging Feature Selection Groups for Classification
# - Extracts TCR counts across samples for each selection group
# - Merges metadata (biological sex & age) with feature selection datasets
# - Sorts columns for consistency across train-test splits
# - Runs H2O AutoML for classification tasks (Age Group & Biological Sex)
##############################################################################

#########################################################
# Extract TCR Counts for Different Feature Selection Groups
# - Reads downsampled TCR data for each selection group
# - Aggregates TCRs across training and testing samples
# - Saves processed datasets for classification
#########################################################

import os
import pandas as pd
from multiprocessing import Pool

# Define source directories
source_dirs = {
    "train": "/dsi/scratch/home/dsi/elihay/downsampled_files/train",
    "test": "/dsi/scratch/home/dsi/elihay/downsampled_files/test"
}
output_dir = "/dsi/scratch/home/dsi/elihay/downsampled_files/dataFiles"

# Function to extract TCR counts from each sample file
def process_file(args):
    file_name, tcrs_of_interest, source_dir = args
    full_path = os.path.join(source_dir, file_name)
    df = pd.read_csv(full_path, delimiter="\t", usecols=['templates', 'amino_acid'])
    
    # Filter by selected TCRs
    df_filtered = df[df['amino_acid'].isin(tcrs_of_interest)]
    sample_id = file_name[:-4]  # Remove '.tsv' extension
    
    # Pivot table to structure data by sample
    pivot_df = df_filtered.pivot_table(index='amino_acid', columns=[lambda x: sample_id], 
                                       values='templates', aggfunc='sum', fill_value=0)
    return pivot_df

# Process groups in parallel
def process_group_tcrs(group, group_data, file_list, source_dir, dataset_type):
    tcrs_of_interest = group_data['amino_acid'].unique().tolist()
    args = [(file, tcrs_of_interest, source_dir) for file in file_list]

    with Pool(processes=60) as pool:
        results = pool.map(process_file, args)

    combined_df = pd.concat(results, axis=1).fillna(0)
    combined_df = combined_df.sort_values(by=combined_df.columns.tolist(), ascending=False)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{group}_{dataset_type}_tcrs_across_samples.csv')
    combined_df.to_csv(output_path)
    
    return f"Data for {group} in {dataset_type} dataset saved to {output_path}"

# Main execution
if __name__ == '__main__':
    input_file_path = "/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/f_final_tcr_selection_adjusted_distribution_with_random_groups.csv"
    df = pd.read_csv(input_file_path)
    
    for dataset_type, source_dir in source_dirs.items():
        file_list = [file for file in os.listdir(source_dir) if file.endswith('.tsv')]
        
        for group in df['selection_group'].unique():
            group_data = df[df['selection_group'] == group]
            result = process_group_tcrs(group, group_data, file_list, source_dir, dataset_type)
            print(result)


#########################################################
# Merge Biological Sex & Age Metadata with Feature Data
# - Transposes dataset so samples are rows
# - Merges metadata for classification
#########################################################

import pandas as pd
import os

def process_and_merge_files(source_dir, metadata_file, output_dir):
    # Load metadata file
    df_metadata = pd.read_excel(metadata_file)
    df_metadata['sample name'] = df_metadata['sample name'].str.replace('.tsv', '')

    os.makedirs(output_dir, exist_ok=True)

    # Process each dataset file
    for file_name in os.listdir(source_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(source_dir, file_name)
            df = pd.read_csv(file_path)

            # Transpose and rename columns
            df_transposed = df.transpose()
            df_transposed.columns = df_transposed.iloc[0]  # Set first row as column headers
            df_transposed = df_transposed[1:].reset_index().rename({'index': 'sample name'}, axis='columns')

            # Merge with metadata
            merged_df = pd.merge(df_transposed, df_metadata, on='sample name', how='inner')

            # Save merged dataset
            output_file_path = os.path.join(output_dir, file_name)
            merged_df.to_csv(output_file_path, index=False)
            print(f"Processed and saved: {output_file_path}")

# Define paths
source_dir = '/dsi/scratch/home/dsi/elihay/downsampled_files/dataFiles/'
metadata_file = '/dsi/scratch/home/dsi/elihay/Matched_File_Data.xlsx'
output_dir = '/dsi/scratch/home/dsi/elihay/downsampled_files/dataFilesAfterMerge/'

process_and_merge_files(source_dir, metadata_file, output_dir)


#########################################################
# Sort Columns by TCR Selection Order
# - Ensures feature consistency across training & test sets
#########################################################

import os
import pandas as pd

# Define directories
source_dir = '/dsi/scratch/home/dsi/elihay/downsampled_files/dataFilesAfterMerge'
output_dir = '/dsi/scratch/home/dsi/elihay/downsampled_files/dataFilesAfterMergeSorted'
reference_file = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/f_final_tcr_selection_adjusted_distribution_with_random_groups.csv'

os.makedirs(output_dir, exist_ok=True)

# Load reference TCR ordering
reference_df = pd.read_csv(reference_file)

# Dictionary to store train-test file pairs
train_files = {}
test_files = {}

# Identify files
for file_name in os.listdir(source_dir):
    if file_name.endswith('.csv'):
        if 'train' in file_name:
            key = file_name.replace('_train_tcrs_across_samples.csv', '')
            train_files[key] = file_name
        elif 'test' in file_name:
            key = file_name.replace('_test_tcrs_across_samples.csv', '')
            test_files[key] = file_name

# Function to sort columns
def sort_and_save(file_path, correct_order, output_filename):
    df = pd.read_csv(file_path)
    
    # Reorder columns while keeping metadata columns intact
    ordered_columns = [col for col in correct_order if col in df.columns]
    remaining_columns = [col for col in df.columns if col not in ordered_columns + ['sample name', 'Biological Sex', 'Age']]
    sorted_df = df[['sample name'] + ordered_columns + remaining_columns + ['Biological Sex', 'Age']]
    
    # Save sorted dataset
    sorted_df.to_csv(os.path.join(output_dir, output_filename), index=False)

# Process each dataset group
for key in train_files:
    correct_order = reference_df[reference_df['selection_group'] == key]['amino_acid'].tolist()
    
    # Process train file
    train_file = os.path.join(source_dir, train_files[key])
    sort_and_save(train_file, correct_order, train_files[key])
    
    # Process test file
    test_file = os.path.join(source_dir, test_files[key])
    sort_and_save(test_file, correct_order, test_files[key])

print("Sorting and saving completed.")
