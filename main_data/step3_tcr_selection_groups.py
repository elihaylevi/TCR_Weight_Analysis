##############################################################################
# STEP 3: Creating TCR Feature Selection Groups
# - Groups TCRs based on their frequency across samples
# - Selects top-performing TCRs based on different clonality metrics
# - Implements both predefined top selections and randomized selections
# - Two versions: with and without publicity threshold
##############################################################################

######################################################
# Create Groups with Publicity Threshold
# - Filters TCRs with sample appearances ≤ 1226
# - Defines range-based groups
# - Selects top 1500 TCRs per clonality metric
# - Generates random selections matching top-group proportions
######################################################

import pandas as pd
import numpy as np

# Load TCR clonality dataset
file_path = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/tcr_counts_and_cloness_with_consistency_area2.csv'
df = pd.read_csv(file_path)

# Filter TCRs that appear in 1226 or fewer samples
df = df[df['sample_appearance_count'] <= 1226]

# Define frequency-based grouping
df['range_group'] = pd.cut(df['sample_appearance_count'], bins=[-1, 225, 725, 1225], labels=['225-rest', '725-226', '1225-726'])

# List of clonality metrics for selection
metrics = ['average_count', 'explosion_area', 'consistency_area', 'consistency_area2', 'variance_score']
top_combined = pd.DataFrame()

# Select the top 1500 TCRs for each metric (allowing duplicates across metrics)
for metric in metrics:
    top_metric = df.nlargest(1500, metric).copy()
    top_metric['selection_group'] = f'{metric}_top'
    top_combined = pd.concat([top_combined, top_metric])

# Compute the proportion of each range group in top selections
range_counts = top_combined['range_group'].value_counts(normalize=True)

# Save the top-selected TCRs
output_path = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/top_combined_with_tags_and_ranges.csv'
top_combined.to_csv(output_path, index=False)
print(f"Top combined selections saved to {output_path}")

# Generate randomized selections maintaining range-group proportions
random_selection_combined = pd.DataFrame()

for i in range(1, 6):  # Create 5 random selection sets
    random_group_tag = f'random_selection_{i}'
    
    random_selections = []
    for range_group, proportion in range_counts.items():
        count = int(1500 * proportion)  # Number of TCRs to select per range
        available_rows = df[(~df.index.isin(top_combined.index)) & (df['range_group'] == range_group)]
        selected = available_rows.sample(n=count, replace=True)  # Allow duplicates in selection
        random_selections.append(selected)
    
    random_group_df = pd.concat(random_selections)
    random_group_df['selection_group'] = random_group_tag
    random_selection_combined = pd.concat([random_selection_combined, random_group_df])

# Combine top-selected and randomly selected groups
final_selection = pd.concat([top_combined, random_selection_combined]).reset_index(drop=True)

# Save the final selection
output_path = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/final_tcr_selection_adjusted_distribution_with_random_groups.csv'
final_selection.to_csv(output_path, index=False)
print(f"Final selection saved to {output_path}. Total rows: {len(final_selection)}")

######################################################
# Create Groups Without Publicity Threshold
# - Uses a higher sample appearance threshold (≤ 2450)
# - Defines new range-based groups
# - Maintains same top-selection and randomization process
# - Includes an additional "publicity" group for most frequent TCRs
######################################################

# Load the updated dataset without publicity filtering
file_path = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/subset123_tcr_counts_and_cloness.csv'
df = pd.read_csv(file_path)

# Define new frequency-based groups
df['range_group'] = pd.cut(df['sample_appearance_count'], bins=[-1, 450, 1450, 2450], labels=['450-rest', '1450-451', '2450-1451'])

# Select the top 1500 TCRs per metric
top_combined = pd.DataFrame()
for metric in metrics:
    top_metric = df.nlargest(1500, metric).copy()
    top_metric['selection_group'] = f'{metric}_top'
    top_metric = top_metric.sort_values(by=[metric], ascending=False)  # Sort within group
    top_combined = pd.concat([top_combined, top_metric])

# Compute range-group proportions in top selections
range_counts = top_combined['range_group'].value_counts(normalize=True)

# Generate randomized selections maintaining range proportions
random_selection_combined = pd.DataFrame()
for i in range(1, 6):  # Create 5 random selection sets
    random_group_tag = f'random_selection_{i}'
    
    random_selections = []
    for range_group, proportion in range_counts.items():
        count = int(1500 * proportion)  # Number of TCRs to select per range
        available_rows = df[(~df.index.isin(top_combined.index)) & (df['range_group'] == range_group)]
        selected = available_rows.sample(n=count, replace=True)  # Allow duplicates in selection
        random_selections.append(selected)
    
    random_group_df = pd.concat(random_selections)
    random_group_df['selection_group'] = random_group_tag
    random_selection_combined = pd.concat([random_selection_combined, random_group_df])

# Add a "publicity" group with the 1500 most frequent TCRs
publicity_top = df.nlargest(1500, 'sample_appearance_count').copy()
publicity_top['selection_group'] = 'publicity_top'
publicity_top = publicity_top.sort_values(by=['sample_appearance_count'], ascending=False)

# Combine top selections, random selections, and publicity group
final_selection = pd.concat([top_combined, random_selection_combined, publicity_top]).reset_index(drop=True)

# Save the final dataset
output_path = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/f_final_tcr_selection_adjusted_distribution_with_random_groups.csv'
final_selection.to_csv(output_path, index=False)
print(f"Final selection saved to {output_path}. Total rows: {len(final_selection)}")
