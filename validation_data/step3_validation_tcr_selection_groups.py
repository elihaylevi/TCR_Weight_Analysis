############################################################
# STEP 3: TCR Selection Groups for Validation Data
# - Group TCRs based on sample appearance count
# - Select top 1500 TCRs by different scoring metrics
# - Add a 'publicity' group for most frequent TCRs
# - Generate randomized control groups matching distributions
############################################################

import pandas as pd
import numpy as np

# Define input and output file paths
INPUT_FILE = '/dsi/scratch/home/dsi/elihay/dean/tcr_counts_and_cloness_with_variance24.csv'
TOP_COMBINED_OUTPUT = '/dsi/scratch/home/dsi/elihay/dean/top_combined_with_tags_and_ranges1500.csv'
FINAL_SELECTION_OUTPUT = '/dsi/scratch/home/dsi/elihay/dean/final_tcr_selection_adjusted_distribution_with_random_groups1500.csv'

# Load the CSV file
df = pd.read_csv(INPUT_FILE)

############################################################
# STEP 1: Define Range Groups Based on Sample Appearance
############################################################

# Retain TCRs appearing in fewer than 279 samples
df = df[df['sample_appearance_count'] < 279]

# Define two groups based on sample appearance counts
df['range_group'] = pd.cut(df['sample_appearance_count'], bins=[-1, 100, 278], labels=['0-100', '101-279'])

############################################################
# STEP 2: Select Top 1500 TCRs Based on Different Metrics
############################################################

# Define ranking metrics
metrics = ['average_count', 'consistency_area', 'variance_score']
top_combined = pd.DataFrame()

# Select the top 1500 TCRs for each metric
for metric in metrics:
    top_metric = df.nlargest(1500, metric).copy()
    top_metric['selection_group'] = f'{metric}_top'
    top_combined = pd.concat([top_combined, top_metric])

# Additional group: Select the top 1500 most frequent TCRs
top_publicity = df.nlargest(1500, 'sample_appearance_count').copy()
top_publicity['selection_group'] = 'publicity_top'
top_combined = pd.concat([top_combined, top_publicity])

# Save top-selected TCRs
top_combined.to_csv(TOP_COMBINED_OUTPUT, index=False)
print(f"Top combined selections saved to {TOP_COMBINED_OUTPUT}")

############################################################
# STEP 3: Generate Randomized Control Groups
############################################################

# Calculate proportion of range groups (excluding publicity group)
range_counts = top_combined[top_combined['selection_group'] != 'publicity_top']['range_group'].value_counts(normalize=True)

# Initialize empty DataFrame for random selections
random_selection_combined = pd.DataFrame()

# Create 5 randomized groups
for i in range(1, 6):
    random_group_tag = f'random_selection_{i}'
    random_selections = []

    # Select random TCRs based on range group proportions
    for range_group, proportion in range_counts.items():
        count = int(1500 * proportion)  # Compute sample count based on proportion
        available_rows = df[(~df.index.isin(top_combined.index)) & (df['range_group'] == range_group)]
        selected = available_rows.sample(n=count, replace=True)  # Allow repetitions
        random_selections.append(selected)

    # Combine selections for this group
    random_group_df = pd.concat(random_selections)
    random_group_df['selection_group'] = random_group_tag
    random_selection_combined = pd.concat([random_selection_combined, random_group_df])

# Combine top-selected TCRs with randomized control groups
final_selection = pd.concat([top_combined, random_selection_combined]).reset_index(drop=True)

# Save final selection
final_selection.to_csv(FINAL_SELECTION_OUTPUT, index=False)
print(f"Final selection saved to {FINAL_SELECTION_OUTPUT}. Total rows: {len(final_selection)}")

print("Step 3 (TCR Selection Groups) complete.")
