import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the baseline methods data
baseline_df = pd.read_csv('clustering_summary.csv')

# Load the proposed model data
proposed_model_df = pd.read_csv('proposed model/clustering_silhouette_scores_summary.csv')

# Get the best results for each dataset from the proposed model
best_proposed_df = pd.read_csv('proposed model/best_clustering_results.csv')

# Map dataset names from proposed model to match baseline dataset names
dataset_mapping = {
    'CustomerPersonality_D1.csv': '1_CustomerPersonality',
    'SupermarketAnalysis_D2.csv': '2_SupermarketAnalysis',
    'CreditCardFraud_C1.csv': '3_CreditCardFraud',
    'MedicalCost_E1.csv': '4_MedicalCost'
}

# Create a new dataframe for comparison
comparison_data = []

# Process baseline methods
for dataset in dataset_mapping.values():
    # Get the best silhouette score for each feature reduction method for this dataset
    for method in ['PCA', 'LDA', 'TSNE', 'IFS']:
        dataset_method_df = baseline_df[(baseline_df['Dataset'] == dataset) & 
                                       (baseline_df['Feature_Reduction_Method'] == method)]
        
        if not dataset_method_df.empty:
            # Find the best silhouette score for this method and dataset
            best_row = dataset_method_df.loc[dataset_method_df['Silhouette_Score'].idxmax()]
            
            # Extract the number of clusters
            if method == 'TSNE':
                selected_features = 'N/A'
            else:
                selected_features = best_row['Selected_Features']
                
            # Handle DBSCAN's special format for clusters
            clusters = best_row['Optimal_Clusters']
            if isinstance(clusters, str) and 'eps=' in clusters:
                # Just count this as 1 parameter set rather than trying to parse
                num_clusters = 1
            else:
                num_clusters = clusters
                
            comparison_data.append({
                'Dataset': dataset,
                'Method': method,
                'Silhouette_Score': best_row['Silhouette_Score'],
                'Num_Clusters': num_clusters,
                'Algorithm': best_row['Clustering_Algorithm']
            })

# Process proposed model
for proposed_dataset, baseline_dataset in dataset_mapping.items():
    # Get the best result for this dataset from the proposed model
    proposed_row = best_proposed_df[best_proposed_df['Dataset Name'] == proposed_dataset].iloc[0]
    
    comparison_data.append({
        'Dataset': baseline_dataset,
        'Method': 'Proposed Model',
        'Silhouette_Score': proposed_row['Silhouette Score'],
        'Num_Clusters': proposed_row['# Clusters'],
        'Algorithm': proposed_row['Algorithm']
    })

# Convert to DataFrame
comparison_df = pd.DataFrame(comparison_data)

# Create visualizations
plt.figure(figsize=(14, 10))

# 1. Bar chart for silhouette scores
plt.subplot(2, 1, 1)
sns.barplot(x='Dataset', y='Silhouette_Score', hue='Method', data=comparison_df)
plt.title('Silhouette Score Comparison: Baseline Methods vs Proposed Model', fontsize=14)
plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Highlight the best method for each dataset
for dataset in dataset_mapping.values():
    dataset_df = comparison_df[comparison_df['Dataset'] == dataset]
    best_method_idx = dataset_df['Silhouette_Score'].idxmax()
    best_method = dataset_df.loc[best_method_idx]
    
    plt.text(x=list(dataset_mapping.values()).index(dataset), 
             y=best_method['Silhouette_Score'] + 0.02,
             s='★ Best',
             ha='center',
             fontweight='bold',
             color='red')

# 2. Bar chart for number of clusters
plt.subplot(2, 1, 2)
sns.barplot(x='Dataset', y='Num_Clusters', hue='Method', data=comparison_df)
plt.title('Number of Clusters Comparison: Baseline Methods vs Proposed Model', fontsize=14)
plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Number of Clusters', fontsize=12)
plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('baseline_vs_proposed_comparison.png', dpi=300, bbox_inches='tight')

# Create a heatmap for silhouette scores
plt.figure(figsize=(12, 8))
# Pivot the data for the heatmap
heatmap_data = comparison_df.pivot(index='Method', columns='Dataset', values='Silhouette_Score')

# Create the heatmap
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
plt.title('Silhouette Score Comparison Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('silhouette_score_heatmap.png', dpi=300, bbox_inches='tight')

# Print summary of best methods per dataset
print("\nBest Method for Each Dataset (by Silhouette Score):")
for dataset in dataset_mapping.values():
    dataset_df = comparison_df[comparison_df['Dataset'] == dataset]
    best_idx = dataset_df['Silhouette_Score'].idxmax()
    best_row = dataset_df.loc[best_idx]
    print(f"{dataset}: {best_row['Method']} (Score: {best_row['Silhouette_Score']:.4f}, Clusters: {best_row['Num_Clusters']})")

plt.show()