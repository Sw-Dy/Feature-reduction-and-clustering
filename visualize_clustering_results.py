import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Load the results
results_df = pd.read_csv('clustering_silhouette_scores_summary.csv')

# Create a figure for the silhouette scores comparison
plt.figure(figsize=(14, 10))

# Group by dataset and algorithm, get the max silhouette score
best_results = results_df.loc[results_df.groupby(['Dataset Name', 'Algorithm'])['Silhouette Score'].idxmax()]

# Create a pivot table for easier plotting
pivot_df = best_results.pivot(index='Dataset Name', columns='Algorithm', values='Silhouette Score')

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
plt.title('Best Silhouette Scores by Dataset and Algorithm', fontsize=16)
plt.tight_layout()
plt.savefig('silhouette_scores_heatmap.png', dpi=300, bbox_inches='tight')

# Plot bar chart comparing algorithms
plt.figure(figsize=(14, 8))

# Get the best algorithm for each dataset
best_algorithm = best_results.loc[best_results.groupby('Dataset Name')['Silhouette Score'].idxmax()]

# Sort by silhouette score
best_algorithm = best_algorithm.sort_values('Silhouette Score', ascending=False)

# Create bar plot
bar = sns.barplot(x='Dataset Name', y='Silhouette Score', hue='Algorithm', data=best_algorithm)

# Customize the plot
plt.title('Best Clustering Algorithm and Silhouette Score by Dataset', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.0)  # Silhouette scores range from -1 to 1
plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Target Score (0.8)')
plt.legend(title='Algorithm', loc='lower right')
plt.tight_layout()
plt.savefig('best_algorithm_by_dataset.png', dpi=300, bbox_inches='tight')

# Create a summary table with the best results
best_results_summary = best_algorithm[['Dataset Name', 'Algorithm', 'Parameters Used', '# Clusters', 'Silhouette Score']]
best_results_summary = best_results_summary.sort_values('Silhouette Score', ascending=False)
best_results_summary.to_csv('best_clustering_results.csv', index=False)

# Create a bar chart showing all algorithms for each dataset
plt.figure(figsize=(16, 10))

# Sort datasets by their best silhouette score
dataset_order = best_algorithm.sort_values('Silhouette Score', ascending=False)['Dataset Name'].tolist()

# Create a categorical type with our custom order
results_df['Dataset Name'] = pd.Categorical(results_df['Dataset Name'], categories=dataset_order, ordered=True)

# Sort the dataframe
results_df = results_df.sort_values('Dataset Name')

# Create grouped bar plot
bar = sns.catplot(x='Dataset Name', y='Silhouette Score', hue='Algorithm', 
                 data=results_df, kind='bar', height=6, aspect=2)

# Customize the plot
plt.title('Silhouette Scores by Dataset and Algorithm', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.ylim(-0.2, 1.0)  # Silhouette scores range from -1 to 1
plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Target Score (0.8)')
plt.legend(title='Algorithm')
plt.tight_layout()
plt.savefig('all_algorithms_comparison.png', dpi=300, bbox_inches='tight')

# Print summary of datasets that achieved the target silhouette score (0.8-0.9)
target_achieved = best_algorithm[best_algorithm['Silhouette Score'] >= 0.8]
print("\nDatasets that achieved the target silhouette score (0.8-0.9):")
print(target_achieved[['Dataset Name', 'Algorithm', 'Silhouette Score']].to_string(index=False))

# Print overall best algorithm
algorithm_performance = results_df.groupby('Algorithm')['Silhouette Score'].mean().sort_values(ascending=False)
print("\nOverall algorithm performance (average silhouette score):")
print(algorithm_performance)

print("\nVisualization complete! Check the generated PNG files for visual results.")