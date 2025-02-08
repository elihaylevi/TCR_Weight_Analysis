########################################################
# PART 3: Classification of TCR Groups using One-Hot Encoded Sequences
# - Converts TCR sequences to fixed-length one-hot vectors
# - Uses H2O AutoML for classification
# - Compares Random Selection 1 vs. Other TCR Groups
# - Saves leaderboard results for the best models
########################################################

import os
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
from sklearn.preprocessing import OneHotEncoder

# Initialize H2O cluster
h2o.init()

# Define output directory for results
output_dir = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/AutoMLResults'
os.makedirs(output_dir, exist_ok=True)

# Load dataset containing selected TCR sequences
file_path = '/dsi/scratch/home/dsi/elihay/downsampled_files/outputs/final_tcr_selection_adjusted_distribution_with_random_groups.csv'
df = pd.read_csv(file_path)

########################################################
# STEP 1: Preprocessing - Sequence Cleaning & Padding
########################################################

# Remove invalid sequences (drop NaN values and sequences with "*")
df = df.dropna(subset=['amino_acid'])
df = df[~df['amino_acid'].str.contains(r'\*')]

# Determine maximum sequence length for padding
max_length = df['amino_acid'].apply(len).max()
print(f"Padding sequences to a fixed length: {max_length}")

def pad_sequences(sequences, max_length):
    """
    Pads TCR sequences to a fixed length using '0' padding.
    """
    return [seq.ljust(max_length, '0') for seq in sequences]

def one_hot_encode_sequences(sequences, max_length):
    """
    One-hot encodes amino acid sequences.
    - Uses 20 standard amino acids + '0' for padding.
    """
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY0')  # Include '0' for padding
    encoder = OneHotEncoder(categories=[amino_acids] * max_length, sparse=False, handle_unknown='ignore')
    split_sequences = [list(seq) for seq in sequences]
    return encoder.fit_transform(split_sequences)

########################################################
# STEP 2: AutoML Training for Group Classification
########################################################

def run_automl(group_name, random_group, other_group):
    """
    Runs H2O AutoML for classification between:
    - Random Selection 1 vs. Another TCR Group
    - Uses one-hot encoded TCR sequences
    - Saves leaderboard of top models
    """

    print(f"\nRunning AutoML for Random Selection 1 vs {group_name}...")

    # Pad sequences to a fixed length
    random_seqs = pad_sequences(random_group['amino_acid'], max_length)
    other_seqs = pad_sequences(other_group['amino_acid'], max_length)

    # One-Hot Encode sequences
    sequences = random_seqs + other_seqs
    X_encoded = one_hot_encode_sequences(sequences, max_length)

    # Assign labels: 1 for random selection, 0 for the other group
    labels = [1] * len(random_seqs) + [0] * len(other_seqs)

    # Create Pandas DataFrame & Convert to H2OFrame
    data = pd.DataFrame(X_encoded)
    data['label'] = labels
    h2o_data = h2o.H2OFrame(data)
    h2o_data['label'] = h2o_data['label'].asfactor()  # Convert label to categorical

    # Train-test split (75% train, 25% test)
    train, test = h2o_data.split_frame(ratios=[0.75], seed=42)
    features = list(h2o_data.columns[:-1])  # All columns except 'label'
    target = 'label'

    try:
        # Run H2O AutoML (Excluding Deep Learning)
        aml = H2OAutoML(max_runtime_secs=1000, seed=42, exclude_algos=["DeepLearning"])
        aml.train(x=features, y=target, training_frame=train, leaderboard_frame=test)

        # Extract the top 5 models from the leaderboard
        leaderboard = aml.leaderboard.as_data_frame().head(5)
        leaderboard = leaderboard[['model_id', 'auc']]  # Keep only model ID and AUC

        # Save results
        results_file = os.path.join(output_dir, f"automl_results_{group_name}.csv")
        leaderboard.to_csv(results_file, index=False)
        print(f"Results saved for group {group_name} at {results_file}")

    except Exception as e:
        print(f"Error during AutoML for {group_name}: {e}")

########################################################
# STEP 3: Run AutoML for Random Selection vs Other Groups
########################################################

# Select **Random Selection 1** as the reference group
random_selection = df[df['selection_group'] == 'random_selection_1']

# Compare Random Selection 1 against every other selection group
other_groups = [group for group in df['selection_group'].unique() if group != 'random_selection_1']

for group_name in other_groups:
    other_group = df[df['selection_group'] == group_name]
    run_automl(group_name, random_selection, other_group)

# Shutdown H2O cluster after processing
h2o.shutdown(prompt=False)

print("\nAutoML classification complete!")
