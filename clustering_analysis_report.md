# Clustering Analysis Report

## Overview

This report summarizes the results of clustering analysis performed on 8 datasets. The goal was to achieve high silhouette scores (preferably 0.8-0.9) through optimal clustering configurations. Three clustering algorithms were evaluated: KMeans, DBSCAN, and Agglomerative Clustering.

## Methodology

1. **Preprocessing**:
   - Standardization using StandardScaler
   - Outlier removal using Isolation Forest (contamination=0.05)
   - One-hot encoding of categorical variables
   - PCA for dimensionality reduction when features > 10 (retaining 95% variance)

2. **Clustering Algorithms**:
   - **KMeans**: Tested k values from 2 to 15 with k-means++ initialization
   - **DBSCAN**: Tested eps values [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] and min_samples values [3, 4, 5, 6, 7, 8, 10, 15]
   - **Agglomerative Clustering**: Tested cluster numbers from 2 to 15 with ward linkage

3. **Evaluation**: Silhouette score calculation for each configuration

## Key Findings

### Best Performing Algorithm

DBSCAN consistently outperformed other algorithms across all datasets, achieving an average silhouette score of 0.824, compared to KMeans (0.427) and Agglomerative Clustering (0.416).

### Datasets Achieving Target Silhouette Scores (≥0.8)

| Dataset Name | Algorithm | Parameters | # Clusters | Silhouette Score |
|--------------|-----------|------------|------------|------------------|
| SupermarketAnalysis_D2.csv | DBSCAN | eps=0.1, min_samples=7 | 2 | 0.994 |
| MedicalCost_E1.csv | DBSCAN | eps=0.1, min_samples=15 | 4 | 0.975 |
| CustomerPersonality_D1.csv | DBSCAN | eps=0.1, min_samples=15 | 5 | 0.896 |
| SupermarketAnalysis_D1.csv | DBSCAN | eps=0.4, min_samples=5 | 4 | 0.894 |
| CustomerPersonality_D3.csv | DBSCAN | eps=0.1, min_samples=15 | 10 | 0.887 |
| CustomerPersonality_D2.csv | DBSCAN | eps=0.15, min_samples=15 | 10 | 0.836 |

### Datasets Not Achieving Target Scores

| Dataset Name | Best Algorithm | Parameters | # Clusters | Silhouette Score |
|--------------|---------------|------------|------------|------------------|
| CreditCardFraud_C1.csv | DBSCAN | eps=0.8, min_samples=10 | 2 | 0.560 |
| CreditCardFraud_C2.csv | DBSCAN | eps=0.8, min_samples=10 | 2 | 0.553 |

## Observations

1. **DBSCAN Superiority**: DBSCAN consistently achieved the highest silhouette scores across all datasets. Its ability to identify clusters of arbitrary shapes and handle noise points made it particularly effective.

2. **Parameter Patterns**:
   - For datasets achieving very high scores (>0.9), smaller eps values (0.1-0.4) were optimal
   - Higher min_samples values (10-15) generally produced better results for most datasets
   - The CreditCardFraud datasets required larger eps values (0.8) to achieve their best scores

3. **Cluster Counts**:
   - Datasets with the highest silhouette scores tended to have fewer clusters (2-5)
   - The CustomerPersonality datasets with more complex structures required more clusters (10)

4. **Dataset Characteristics**:
   - SupermarketAnalysis and MedicalCost datasets showed the clearest cluster structures
   - CreditCardFraud datasets had more challenging cluster structures, resulting in lower silhouette scores

## Recommendations

1. **Algorithm Selection**: DBSCAN is recommended as the primary clustering algorithm for these types of datasets due to its superior performance.

2. **Parameter Tuning**:
   - Start with smaller eps values (0.1-0.4) and higher min_samples (10-15) for DBSCAN
   - Adjust eps based on dataset density (higher for sparser data)

3. **Further Improvements**:
   - For datasets not achieving target scores (CreditCardFraud), consider:
     - Additional feature engineering
     - Testing other clustering algorithms (e.g., OPTICS, Spectral Clustering)
     - Exploring different preprocessing techniques

## Conclusion

The clustering analysis successfully identified optimal configurations for 6 out of 8 datasets, achieving silhouette scores above 0.8. DBSCAN emerged as the most effective algorithm across all datasets, demonstrating its versatility in identifying meaningful clusters in various data structures.