 Clustering Analysis Project

## Overview
This project performs clustering analysis on various datasets, evaluates the performance of different clustering algorithms, and identifies the best-performing datasets and algorithms based on silhouette scores.

## Files and Scripts

### Main Scripts
- `clustering_analysis.py`: Performs clustering analysis on multiple datasets using KMeans, DBSCAN, and Agglomerative Clustering algorithms.
- `visualize_clustering_results.py`: Generates visualizations of clustering results including heatmaps, bar charts, and comparison tables.
- `extract_best_datasets.py`: Identifies the best-performing dataset versions based on silhouette scores.

### Output Files
- `clustering_silhouette_scores_summary.csv`: Contains detailed results of all clustering analyses.
- `best_clustering_results.csv`: Contains the best results for each dataset and algorithm combination.
- `best_datasets_by_silhouette.csv`: Lists the best-performing dataset versions with their scores, algorithms, parameters, and cluster counts.
- `dataset_comparison_report_[timestamp].txt`: Detailed report comparing all dataset versions and their performance.
- `best_datasets_by_silhouette_[timestamp].csv`: Timestamped version of the best datasets file.

### Visualization Files
- `silhouette_scores_heatmap.png`: Heatmap of silhouette scores across datasets and algorithms.
- `all_algorithms_comparison.png`: Bar chart comparing algorithm performance across all datasets.
- `best_algorithm_by_dataset.png`: Bar chart showing the best algorithm for each dataset.

## Usage

### Running the Clustering Analysis
```bash
python clustering_analysis.py
```

### Generating Visualizations
```bash
python visualize_clustering_results.py
```

### Extracting Best Datasets
```bash
python extract_best_datasets.py
```

## Results
The analysis identified the following best-performing datasets:
- CreditCardFraud_C1.csv (Score: 0.5596, Algorithm: DBSCAN)
- CustomerPersonality_D1.csv (Score: 0.8959, Algorithm: DBSCAN)
- MedicalCost_E1.csv (Score: 0.9747, Algorithm: DBSCAN)
- SupermarketAnalysis_D2.csv (Score: 0.9937, Algorithm: DBSCAN)

DBSCAN consistently outperformed other algorithms across all datasets, achieving an average silhouette score of 0.824.