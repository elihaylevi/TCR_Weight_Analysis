

#######################################################
# STEP 1: Prepare Data - Extract, Filter, and Downsample
# - Extract raw TCR files and create metadata
# - Remove files with fewer than 200K TCRs
# - Downsample large files while preserving clonality
#######################################################

#######################################################
# Train-Test Split (75% Train / 25% Test)
# - Randomly shuffle all downsampled files
# - Assign 75% of files to the training set
# - Assign 25% of files to the testing set
# - Move files to corresponding directories
#######################################################

import os
import shutil
import numpy as np

# Define directory paths
source_dir = '/dsi/scratch/home/dsi/elihay/downsampled_files/'
train_dir = '/dsi/scratch/home/dsi/elihay/train/'
test_dir = '/dsi/scratch/home/dsi/elihay/test/'

# Ensure the train and test directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all available files in the source directory
all_files = os.listdir(source_dir)

# Set random seed for reproducibility
random_state = 42
np.random.seed(random_state)

# Shuffle and split files (75% train, 25% test)
np.random.shuffle(all_files)
split_index = int(0.75 * len(all_files))
train_files = all_files[:split_index]
test_files = all_files[split_index:]

# Move files to corresponding directories
for file_name in train_files:
    shutil.move(os.path.join(source_dir, file_name), os.path.join(train_dir, file_name))
    print(f"Moved to train: {file_name}")

for file_name in test_files:
    shutil.move(os.path.join(source_dir, file_name), os.path.join(test_dir, file_name))
    print(f"Moved to test: {file_name}")

