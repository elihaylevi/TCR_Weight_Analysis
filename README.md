# TCR Weight Analysis Repository

## Overview

We introduce **TCR Weight**, a novel clonality-based metric that highlights TCRs with unique clonal behaviors, providing a different perspective for repertoire analysis. TCR Weight integrates seamlessly into machine learning frameworks, enhancing feature prioritization in a manner that we demonstrate to improve classification accuracy. This approach holds potential for robust immune profiling in noisy datasets. TCR Weight enhances the functionality of the TCR repertoire and creates new opportunities for exploring immune responses.

## Repository Structure

```
├── main_data/          # Contains code for data processing and classification
│   ├── step1_data_preprocessing.py
│   ├── step2_calculate_tcr_weight.py
│   ├── step3_tcr_selection_groups.py
│   ├── step4_feature_extraction.py
│   ├── step5a_sex_classification_automl.py
│   ├── step5b_age_classification_automl.py
│   ├── step5c_tcr_sequence_classification.py
│
├── validation_data/    # Contains validation dataset processing scripts
│   ├── step1_validation_data_preprocessing.py
│   ├── step2_validation_calculate_tcr_weight.py
│   ├── step3_validation_tcr_selection_groups.py
│
├── notebooks/          # Jupyter Notebooks for additional analysis
│   ├── TCR_weight_plots.ipynb  # Generates all plots for the results section
│
├── results/            # Contains final figures and output results
│   ├── figures/
│       ├── plots_amino_acid_presence/
│       ├── age_classification_accuracy_heatmap.png
│       ├── age_classification_radar_chart.png
│       ├── biological_sex_classification_accuracy_heatmap.png
│       ├── cdr3_comparison_tcr_weight_vs_publicity.png
│       ├── euclidean_distance_heatmap.png
│       ├── one_hot_classification_auc.png
│
├── requirements.txt    # List of dependencies for the project
└── README.md           # Project documentation (this file)
```

## Installation & Setup

### Prerequisites

Make sure you have Python installed (preferably Python 3.8 or higher). You will also need `pip` to install dependencies.

### Install Dependencies

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

Alternatively, if using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Running Data Preprocessing & Classification

- The `main_data/` folder contains scripts for data preprocessing, feature extraction, and classification.
- The `validation_data/` folder is used for validating findings but is **not used** for classification.
- The scripts in `notebooks/` process and analyze the data.

### Generating Plots for the Results Section

To generate the figures used in the study, open and run `notebooks/TCR_weight_plots.ipynb`. This notebook produces all visualizations found in the results section of the article.

## Figures & Naming Convention

### Key Figures in `results/`

- **`one_hot_classification_auc.png`** - Results of one-hot classification between different TCR selection groups.
- **`cdr3_comparison_tcr_weight_vs_publicity.png`** - Comparison of CDR3 differences between TCR Weight and Publicity.
- **`age_classification_accuracy_heatmap.png`** - Heatmap showing classification accuracy across age groups.
- **`biological_sex_classification_accuracy_heatmap.png`** - Heatmap showing classification accuracy for biological sex.
- **`euclidean_distance_heatmap.png`** - Heatmap of Euclidean distances between TCR selection groups.
- **`age_classification_radar_chart.png`** - Radar chart visualizing age classification accuracy.

## Contact

For questions or collaborations, reach out to [**elihaylevi9@gmail.com**](mailto\:elihaylevi9@gmail.com).

