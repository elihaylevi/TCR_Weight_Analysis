#############################################
# PART 2: Biological Sex Classification - AutoML
# - Uses H2O AutoML to classify **Biological Sex** (Male vs. Female)
# - Runs classification for different feature subset sizes
# - Saves performance metrics and feature selection results
#############################################

import os
import h2o
import pandas as pd
from h2o.automl import H2OAutoML

# Initialize H2O cluster (allocate 70 threads for processing)
h2o.init(nthreads=70)

# Define files to **skip** from classification
skip_files = [
    'random_selection_4_train_tcrs_across_samples.csv',
    'random_selection_4_test_tcrs_across_samples.csv',
    'random_selection_5_train_tcrs_across_samples.csv',
    'random_selection_5_test_tcrs_across_samples.csv'
]

def process_single_file_pair(train_file, test_file, source_dir, output_file):
    """
    Trains an H2O AutoML model to classify biological sex using TCR data.

    - Loads train & test data
    - Runs AutoML on selected feature subsets
    - Saves classification results to CSV
    """
    try:
        # Load training data
        train_df = h2o.import_file(os.path.join(source_dir, train_file))

        # Drop rows with missing 'Biological Sex' and convert it to categorical
        train_df = train_df[train_df['Biological Sex'].isna() == False]
        train_df['Biological Sex'] = train_df['Biological Sex'].asfactor()

        # Prepare selected training dataset
        df_train_selected = train_df

        # Load and preprocess test data
        test_df = h2o.import_file(os.path.join(source_dir, test_file))
        test_df = test_df[test_df['Biological Sex'].isna() == False]
        test_df['Biological Sex'] = test_df['Biological Sex'].asfactor()
        df_test_selected = test_df

        # Define feature selection sizes
        excluded_columns = ['sample name', 'Age', 'Biological Sex']
        max_features = len(df_train_selected.columns) - len(excluded_columns)
        feature_sizes = [25, 50, 100, 250, 500, 750, 1000, 1500]

        results_list = []

        for num_features in feature_sizes:
            if num_features == 1500 and max_features < 1500:
                feature_columns = [col for col in df_train_selected.columns if col not in excluded_columns]
            else:
                feature_columns = [col for col in df_train_selected.columns if col not in excluded_columns][:num_features]

            # Check if all selected features exist in test set
            missing_in_test = [col for col in feature_columns if col not in df_test_selected.columns]
            if missing_in_test:
                print(f"Missing features in test data: {missing_in_test}")
                continue

            X_train_subset = df_train_selected[:, feature_columns + ['Biological Sex']]
            X_test_subset = df_test_selected[:, feature_columns + ['Biological Sex']]

            # Run H2O AutoML with **XGBoost only**
            aml = H2OAutoML(max_runtime_secs=2000, seed=42,
                            project_name=f"automl_binary_classification_{train_file}_size_{num_features}",
                            include_algos=["XGBoost"])
            aml.train(y='Biological Sex', training_frame=X_train_subset)

            # Save leaderboard
            lb = aml.leaderboard
            lb_path = os.path.join(source_dir, f'leaderboard_{train_file}_size_{num_features}_.csv')
            h2o.export_file(lb, path=lb_path, force=True)

            # Get model performance & parameters
            best_model = aml.leader
            best_perf = best_model.model_performance(X_test_subset)
            best_params = {param: best_model.params[param]['actual'] for param in best_model.params}

            # Get number of features selected by best model
            varimp_df = best_model.varimp(use_pandas=True)
            selected_num_features = len(varimp_df[varimp_df['percentage'] > 0])  # Count non-zero importance features

            # Save results
            results = {
                'file': train_file,
                'original_num_features': num_features,
                'selected_num_features': selected_num_features,
                'accuracy': best_perf.accuracy()[0][1],
                'precision': best_perf.precision()[0][1],
                'recall': best_perf.recall()[0][1],
                'f1_score': best_perf.F1()[0][1],
                'roc_auc': best_perf.auc(),
                'Model_ID': best_model.model_id,
                'Hyperparameters': best_params
            }

            results_list.append(results)

            print(f"Results for {train_file} with top {num_features} features:")
            print(results)

        # Save results to output file
        if results_list:
            results_df = pd.DataFrame(results_list)
            if not os.path.exists(output_file):
                results_df.to_csv(output_file, index=False)
            else:
                results_df.to_csv(output_file, mode='a', header=False, index=False)

    except Exception as e:
        print(f"An error occurred: {e}")

def process_files_for_classification(source_dir, output_file):
    """
    Iterates through train-test pairs and runs AutoML classification for Biological Sex.
    """
    train_files = {}
    test_files = {}

    # Identify train & test files
    for file_name in os.listdir(source_dir):
        if file_name in skip_files:
            continue
        if file_name.endswith('.csv'):
            if 'train' in file_name:
                key = file_name.replace('_train_tcrs_across_samples.csv', '')
                train_files[key] = file_name
            elif 'test' in file_name:
                key = file_name.replace('_test_tcrs_across_samples.csv', '')
                test_files[key] = file_name

    for key in train_files:
        train_file = train_files.get(key)
        test_file = test_files.get(key)
        if train_file and test_file:
            process_single_file_pair(train_file, test_file, source_dir, output_file)

# Define paths
source_dir = '/dsi/scratch/home/dsi/elihay/downsampled_files/dataFilesAfterMergeSorted/'
output_file = '/dsi/scratch/home/dsi/elihay/downsampled_files/model_results_binary_sex_classification.csv'

# Run classification on all train-test pairs
process_files_for_classification(source_dir, output_file)

# Shutdown H2O cluster after completion
h2o.cluster().shutdown()

