# Feature Selection and Dimensionality Reduction Tool

This tool provides a comprehensive analysis of datasets using various feature selection and dimensionality reduction techniques.

## Features

- **Data Preprocessing**:
  - Automatic detection of categorical and numerical features
  - One-hot encoding for categorical features
  - Standardization of numerical features

- **Dimensionality Reduction Techniques**:
  - **PCA**: Reduces dimensions while preserving variance
  - **LDA**: Supervised dimensionality reduction using class labels
  - **t-SNE**: Non-linear dimensionality reduction for visualization
  - **Incremental Feature Selection (IFS)**: Ranks features by importance

## Usage

```bash
python feature_selection.py --file [CSV_FILE_PATH] [OPTIONS]
```

### Required Arguments

- `--file`: Path to the CSV dataset file

### Optional Arguments

- `--variance`: Variance threshold for PCA (default: 0.9)
- `--clusters`: Number of clusters for KMeans when generating pseudo-labels (default: 3)
- `--method`: Feature ranking method for IFS, either 'f_test' or 'mutual_info' (default: 'f_test')

## Examples

```bash
# Basic usage with default parameters
python feature_selection.py --file d:\data2\1_CustomerPersonality.csv

# Specify variance threshold for PCA
python feature_selection.py --file d:\data2\2_SupermarketAnalysis.csv --variance 0.95

# Use mutual information for feature ranking
python feature_selection.py --file d:\data2\3_CreditCardFraud.csv --method mutual_info

# Specify number of clusters for pseudo-labels
python feature_selection.py --file d:\data2\4_MedicalCost.csv --clusters 5
```

## Output

The script provides detailed output for each method:

- **PCA**:
  - Number of components explaining the specified variance
  - Top features with highest loadings for each component

- **LDA**:
  - Top features with highest coefficients for each component

- **t-SNE**:
  - Transformed matrix (primarily for visualization)

- **IFS**:
  - Top-k features for k = 5, 10, 15 ranked by importance

## Available Datasets

1. `1_CustomerPersonality.csv`: Customer personality analysis dataset
2. `2_SupermarketAnalysis.csv`: Supermarket sales analysis dataset
3. `3_CreditCardFraud.csv`: Credit card fraud detection dataset
4. `4_MedicalCost.csv`: Medical cost prediction dataset