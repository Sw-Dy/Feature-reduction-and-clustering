import pandas as pd
import datetime

# Get current timestamp for file naming
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Read the clustering results CSV
print("Reading clustering results from clustering_silhouette_scores_summary.csv...")
df = pd.read_csv('clustering_silhouette_scores_summary.csv')

# Extract the base dataset name (before the underscore and version)
def extract_base_name(dataset_name):
    """
    Extract the base name of a dataset (part before the underscore and version)
    
    Args:
        dataset_name (str): The full dataset name (e.g., 'CustomerPersonality_D1.csv')
        
    Returns:
        str: The base name (e.g., 'CustomerPersonality')
    """
    # Split by underscore and take the first part
    parts = dataset_name.split('_')
    return parts[0]

# Add a column for the base dataset name
df['Base Dataset'] = df['Dataset Name'].apply(extract_base_name)
print(f"Found {len(df['Base Dataset'].unique())} unique base datasets")


# Create a detailed report file with timestamp
report_filename = f'dataset_comparison_report_{timestamp}.txt'
print(f"Generating detailed comparison report to {report_filename}...")

with open(report_filename, 'w') as f:
    f.write("=================================================================\n")
    f.write("             CLUSTERING ANALYSIS - DATASET COMPARISON           \n")
    f.write("=================================================================\n")
    f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("All datasets with their best silhouette scores by algorithm:\n")
    f.write("=================================================================\n")
    
    for base_name in sorted(df['Base Dataset'].unique()):
        f.write(f"\n{base_name}:\n")
        f.write("-" * len(base_name) + "\n")
        base_df = df[df['Base Dataset'] == base_name]
        
        # Get the best score for each dataset version (across all algorithms)
        best_by_dataset = base_df.loc[base_df.groupby('Dataset Name')['Silhouette Score'].idxmax()]
        
        # Sort by silhouette score in descending order
        best_by_dataset = best_by_dataset.sort_values('Silhouette Score', ascending=False)
        
        for _, row in best_by_dataset.iterrows():
            f.write(f"  - {row['Dataset Name']}:\n")
            f.write(f"      Score: {row['Silhouette Score']:.4f}\n")
            f.write(f"      Algorithm: {row['Algorithm']}\n")
            f.write(f"      Parameters: {row['Parameters Used']}\n")
            f.write(f"      Clusters: {row['# Clusters']}\n")

# For each base dataset, find the version with the highest silhouette score across all algorithms
print("Identifying best dataset version for each base dataset...")
best_base_datasets = df.loc[df.groupby('Base Dataset')['Silhouette Score'].idxmax()]

# Write the selected datasets to the report
with open(report_filename, 'a') as f:
    f.write("\n=================================================================\n")
    f.write("SELECTED BEST DATASETS\n")
    f.write("=================================================================\n")
    for _, row in best_base_datasets.iterrows():
        f.write(f"- {row['Dataset Name']}:\n")
        f.write(f"  Score: {row['Silhouette Score']:.4f}\n")
        f.write(f"  Algorithm: {row['Algorithm']}\n")
        f.write(f"  Parameters: {row['Parameters Used']}\n")
        f.write(f"  Clusters: {row['# Clusters']}\n\n")

# Select the Dataset Name, Algorithm, Parameters Used, Silhouette Score, and # Clusters columns for the final output
best_datasets_output = best_base_datasets[['Dataset Name', 'Algorithm', 'Parameters Used', 'Silhouette Score', '# Clusters']]

# Rename columns for clarity
best_datasets_output = best_datasets_output.rename(columns={
    'Silhouette Score': 'Best Silhouette Score', 
    '# Clusters': 'Optimum Clusters', 
    'Algorithm': 'Best Algorithm',
    'Parameters Used': 'Algorithm Parameters'
})

# Save to a new CSV file with timestamp
csv_filename = f'best_datasets_by_silhouette_{timestamp}.csv'
best_datasets_output.to_csv(csv_filename, index=False)

# Also save to the standard filename for compatibility
best_datasets_output.to_csv('best_datasets_by_silhouette.csv', index=False)

# Print summary information
print(f"\nBest datasets identified:")
for _, row in best_base_datasets.iterrows():
    print(f"- {row['Dataset Name']} (Score: {row['Silhouette Score']:.4f}, Algorithm: {row['Algorithm']})")

print(f"\nDetailed comparison saved to {report_filename}")
print(f"Best datasets extracted and saved to:")
print(f"- best_datasets_by_silhouette.csv")
print(f"- {csv_filename}")