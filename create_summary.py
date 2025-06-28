import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the datasets and reduction methods
datasets = [
    '1_CustomerPersonality',
    '2_SupermarketAnalysis',
    '3_CreditCardFraud',
    '4_MedicalCost'
]

reduction_methods = ['pca', 'lda', 'tsne', 'ifs']

# Base directory for results
results_dir = 'd:\\data2\\results'

# Function to get the top features for a dataset and method
def get_top_features(dataset, method, top_n=5):
    # Load feature rankings
    rankings_path = os.path.join(results_dir, dataset, f'{dataset}_feature_rankings.csv')
    if not os.path.exists(rankings_path):
        return 'N/A'
    
    rankings = pd.read_csv(rankings_path)
    
    # Filter by method and get top features
    method_rankings = rankings[rankings['Method'] == method.upper()]
    if method_rankings.empty:
        return 'N/A'
    
    # Get top N features
    top_features = method_rankings.sort_values('Rank').head(top_n)['Feature'].tolist()
    return ', '.join(top_features)

# Create a DataFrame to store the combined results
results_data = []

# Process each dataset
for dataset in datasets:
    # Load optimal clustering results
    optimal_path = os.path.join(results_dir, dataset, 'clustering', f'{dataset}_optimal_clustering.csv')
    if not os.path.exists(optimal_path):
        print(f"Warning: No optimal clustering results found for {dataset}")
        continue
    
    optimal_results = pd.read_csv(optimal_path)
    
    # Process each reduction method
    for method in reduction_methods:
        # Get method results
        method_results = optimal_results[optimal_results['Reduction_Method'] == method]
        if method_results.empty:
            print(f"Warning: No results for {method} in {dataset}")
            continue
        
        # Get selected features (only for PCA, LDA, and IFS)
        selected_features = 'N/A'
        if method != 'tsne':
            selected_features = get_top_features(dataset, method)
        
        # Process each clustering algorithm
        for _, row in method_results.iterrows():
            algorithm = row['Clustering_Algorithm']
            silhouette = row['Silhouette_Score']
            
            # Determine optimal number of clusters
            if algorithm == 'kmeans':
                n_clusters = row['Optimal_k']
            elif algorithm == 'hierarchical':
                n_clusters = row['Optimal_n_clusters']
            elif algorithm == 'dbscan':
                n_clusters = f"eps={row['Optimal_eps']}, min_samples={row['Optimal_min_samples']}"
            else:
                n_clusters = 'N/A'
            
            # Add to results
            results_data.append({
                'Dataset': dataset,
                'Feature_Reduction_Method': method.upper(),
                'Selected_Features': selected_features,
                'Clustering_Algorithm': algorithm,
                'Optimal_Clusters': n_clusters,
                'Silhouette_Score': silhouette
            })

# Create the summary DataFrame
summary_df = pd.DataFrame(results_data)

# Save to CSV
summary_path = 'd:\\data2\\clustering_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f"Summary table saved to: {summary_path}")

# Create silhouette score comparison plot
plt.figure(figsize=(14, 10))

# Plot silhouette scores by dataset and reduction method
sns.barplot(x='Feature_Reduction_Method', y='Silhouette_Score', hue='Clustering_Algorithm', 
            data=summary_df, palette='viridis')

plt.title('Silhouette Scores by Reduction Method and Clustering Algorithm', fontsize=16)
plt.xlabel('Feature Reduction Method', fontsize=14)
plt.ylabel('Silhouette Score', fontsize=14)
plt.legend(title='Clustering Algorithm', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plot_path = 'd:\\data2\\silhouette_scores_comparison.png'
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Silhouette score comparison plot saved to: {plot_path}")

# Create dataset comparison plot
plt.figure(figsize=(16, 10))

# Plot silhouette scores by dataset and reduction method
sns.boxplot(x='Dataset', y='Silhouette_Score', hue='Feature_Reduction_Method', 
           data=summary_df, palette='Set2')

plt.title('Silhouette Scores by Dataset and Reduction Method', fontsize=16)
plt.xlabel('Dataset', fontsize=14)
plt.ylabel('Silhouette Score', fontsize=14)
plt.legend(title='Reduction Method', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
plot_path = 'd:\\data2\\dataset_comparison.png'
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Dataset comparison plot saved to: {plot_path}")

# Create heatmap of silhouette scores
pivot_df = summary_df.pivot_table(
    index=['Dataset', 'Clustering_Algorithm'],
    columns='Feature_Reduction_Method',
    values='Silhouette_Score'
)

plt.figure(figsize=(12, 10))
sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
plt.title('Silhouette Scores Heatmap', fontsize=16)
plt.tight_layout()

# Save the heatmap
heatmap_path = 'd:\\data2\\silhouette_heatmap.png'
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"Silhouette score heatmap saved to: {heatmap_path}")

print("\nSummary statistics:")
print(summary_df.groupby(['Feature_Reduction_Method', 'Clustering_Algorithm'])['Silhouette_Score'].mean().sort_values(ascending=False))