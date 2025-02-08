############################################################
# STEP 1: Extract and Preprocess Validation Data
# - Extract files from ZIP (if needed)
# - Remove unnecessary columns
# - Generate metadata for samples
# - Filter files >200K templates
# - Downsample large files to 200K entries
############################################################

import os
import zipfile
import pandas as pd
import shutil
import numpy as np

# Define input/output paths
RAW_ZIP_PATH = '/dsi/scratch/home/dsi/elihay/dean/dean-2015-genomemed.zip'
EXTRACTED_DIR = '/dsi/scratch/home/dsi/elihay/dean_'
CLEANED_DIR = EXTRACTED_DIR  # Same directory after cleaning
METADATA_FILE = '/dsi/scratch/home/dsi/elihay/dean/dean_metadata.csv'
DOWNSAMPLED_DIR = '/dsi/scratch/home/dsi/elihay/dean/downsampled_files'

# Ensure directories exist
os.makedirs(EXTRACTED_DIR, exist_ok=True)
os.makedirs(DOWNSAMPLED_DIR, exist_ok=True)

############################################################
# STEP 1: Extract Files from ZIP 
############################################################

if os.path.exists(RAW_ZIP_PATH):
    with zipfile.ZipFile(RAW_ZIP_PATH, 'r') as zip_ref:
        for file_name in zip_ref.namelist():
            output_file_path = os.path.join(EXTRACTED_DIR, file_name)
            if not os.path.exists(output_file_path):
                zip_ref.extract(file_name, EXTRACTED_DIR)
                print(f"Extracted: {file_name}")
            else:
                print(f"Skipped (already exists): {file_name}")

############################################################
# STEP 2: Clean Files (Keep Relevant Columns)
############################################################

# Define columns to retain
COLUMNS_TO_KEEP = ['sample_catalog_tags', 'rearrangement', 'amino_acid', 'frame_type', 'templates']

for filename in os.listdir(EXTRACTED_DIR):
    if filename.endswith('.tsv'):
        file_path = os.path.join(EXTRACTED_DIR, filename)
        try:
            df = pd.read_csv(file_path, sep='\t', low_memory=False)
            df = df[COLUMNS_TO_KEEP]  # Keep only required columns
            df.to_csv(file_path, sep='\t', index=False)  # Overwrite file
            print(f"Cleaned and saved: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

############################################################
# STEP 3: Generate Metadata for Samples
############################################################

metadata = []

for filename in os.listdir(EXTRACTED_DIR):
    if filename.endswith('.tsv'):
        file_path = os.path.join(EXTRACTED_DIR, filename)
        try:
            df = pd.read_csv(file_path, sep='\t', low_memory=False)

            # Extract sample catalog tag (if available)
            sample_catalog_tags = df['sample_catalog_tags'].iloc[0] if 'sample_catalog_tags' in df.columns else ''

            # Calculate total templates for frame_type 'In'
            total_templates = df.loc[df['frame_type'] == 'In', 'templates'].sum() if 'templates' in df.columns else 0

            # Append metadata
            metadata.append({
                'Sample Name': filename,
                'Sample Catalog Tags': sample_catalog_tags,
                'Total Templates': total_templates
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save metadata as CSV
metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(METADATA_FILE, index=False)
print(f"Metadata saved to {METADATA_FILE}")

############################################################
# STEP 4: Move Files >200K Templates to Downsampling Folder
############################################################

# Load metadata
metadata_df = pd.read_csv(METADATA_FILE)

# Filter for large files
large_files = metadata_df[metadata_df['Total Templates'] > 200000]['Sample Name']

for filename in large_files:
    src_path = os.path.join(EXTRACTED_DIR, filename)
    dst_path = os.path.join(DOWNSAMPLED_DIR, filename)

    try:
        shutil.move(src_path, dst_path)
        print(f"Moved {filename} to {DOWNSAMPLED_DIR} for downsampling")
    except Exception as e:
        print(f"Error moving {filename}: {e}")

############################################################
# STEP 5: Downsample Large Files (Limit: 200K Entries)
# - Expands sequences based on 'templates' values
# - Randomly samples 200,000 rows if file is too large
# - Groups by rearrangement and sums template counts
############################################################

for filename in os.listdir(DOWNSAMPLED_DIR):
    if filename.endswith('.tsv'):
        file_path = os.path.join(DOWNSAMPLED_DIR, filename)
        try:
            df = pd.read_csv(file_path, sep='\t')

            # Step 1: Keep only 'In' frame_type
            df = df[df['frame_type'] == 'In']

            # Step 2: Expand rows based on 'templates' value
            expanded_rows = []
            for _, row in df.iterrows():
                expanded_rows.extend([row] * int(row['templates']))  # Repeat each row
            expanded_df = pd.DataFrame(expanded_rows).assign(templates=1)  # Set templates to 1

            # Step 3: Randomly sample 200,000 rows (if needed)
            if len(expanded_df) > 200000:
                expanded_df = expanded_df.sample(200000, random_state=42)

            # Step 4: Aggregate by rearrangement
            downsampled_df = expanded_df.groupby('rearrangement', as_index=False).agg({'templates': 'sum'})

            # Step 5: Merge with original metadata to retain amino acid & sample tags
            downsampled_df = downsampled_df.merge(
                df[['rearrangement', 'sample_catalog_tags', 'amino_acid', 'frame_type']].drop_duplicates('rearrangement'),
                on='rearrangement',
                how='left'
            )

            # Overwrite original file with downsampled version
            downsampled_df.to_csv(file_path, sep='\t', index=False)
            print(f"Downsampled and saved: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Step 1 (Validation Data Preprocessing) complete.")
