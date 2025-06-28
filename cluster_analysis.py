import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Define the datasets and reduction methods
datasets = [
    '1_CustomerPersonality',
    '2_SupermarketAnalysis',
    '3_CreditCardFraud',
    '4_MedicalCost'
]

reduction_methods = ['pca', 'lda', 'tsne', 'ifs']

# Define clustering parameters to try
kmeans_k_values = list(range(2, 11))  # 2 to 10 clusters
hierarchical_n_clusters = list(range(2, 11))  # 2 to 10 clusters
dbscan_eps_values = [0.3, 0.5, 0.7, 1.0]
dbscan_min_samples_values = [3, 5, 10]


def load_reduced_data(dataset_name, results_dir='results'):
    """Load the reduced data for a dataset by reprocessing the original data."""
    # Create a dictionary to store reduced data
    reduced_data = {}
    
    # Load the original dataset
    original_data_path = f"d:\\data2\\{dataset_name}.csv"
    if not os.path.exists(original_data_path):
        print(f"Error: Original dataset {original_data_path} not found.")
        return None
    
    # Import necessary functions from process_all_datasets.py
    from process_all_datasets import load_and_preprocess_data, apply_pca, apply_lda, apply_tsne, apply_ifs
    
    # Load and preprocess data
    X, target, feature_names, original_columns = load_and_preprocess_data(original_data_path)
    print(f"Dataset {dataset_name} loaded and preprocessed. Shape: {X.shape}")
    
    # Apply PCA
    X_pca, n_components, pca_top_features = apply_pca(X, feature_names)
    reduced_data['pca'] = X_pca
    print(f"PCA applied with {n_components} components")
    
    # Apply LDA
    X_lda, lda_features, lda_top_features = apply_lda(X, feature_names, target)
    reduced_data['lda'] = X_lda
    print(f"LDA applied with {X_lda.shape[1]} components")
    
    # Apply t-SNE
    X_tsne = apply_tsne(X)
    reduced_data['tsne'] = X_tsne
    print(f"t-SNE applied with 2 components")
    
    # Apply IFS and select top features
    ifs_features, ifs_top_features = apply_ifs(X, feature_names, target)
    
    # For IFS, we need to select the top features from the original data
    # Get the top 10 feature names
    top_ifs_features = [feature for feature, _ in ifs_features[:10]]
    
    # Find indices of these features in the feature_names list
    top_indices = []
    for feature in top_ifs_features:
        try:
            idx = feature_names.index(feature)
            top_indices.append(idx)
        except ValueError:
            print(f"Warning: Feature {feature} not found in feature_names")
    
    # Select these features from X
    if top_indices:
        X_ifs = X[:, top_indices]
        reduced_data['ifs'] = X_ifs
        print(f"IFS applied with {X_ifs.shape[1]} features")
    else:
        # If no features were found, use PCA data as fallback
        reduced_data['ifs'] = X_pca
        print("Warning: No IFS features found, using PCA data instead")
    
    return reduced_data


def apply_kmeans(X, k_values):
    """Apply KMeans clustering with different k values and return the best result."""
    best_score = -1
    best_k = None
    best_labels = None
    
    results = []
    
    for k in k_values:
        try:
            # Ensure k is not greater than the number of samples
            if k >= X.shape[0]:
                print(f"Skipping k={k} as it's >= number of samples ({X.shape[0]})")
                continue
                
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Calculate silhouette score
            if len(np.unique(labels)) > 1:  # Silhouette score requires at least 2 clusters
                score = silhouette_score(X, labels)
                results.append({'k': k, 'score': score})
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
        except Exception as e:
            print(f"Error with KMeans (k={k}): {str(e)}")
    
    return best_k, best_score, best_labels, results


def apply_hierarchical(X, n_clusters_values):
    """Apply Hierarchical (Agglomerative) clustering with different numbers of clusters and return the best result."""
    best_score = -1
    best_n_clusters = None
    best_labels = None
    
    results = []
    
    for n_clusters in n_clusters_values:
        try:
            # Ensure n_clusters is not greater than the number of samples
            if n_clusters >= X.shape[0]:
                print(f"Skipping n_clusters={n_clusters} as it's >= number of samples ({X.shape[0]})")
                continue
                
            hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
            labels = hierarchical.fit_predict(X)
            
            # Calculate silhouette score
            if len(np.unique(labels)) > 1:  # Silhouette score requires at least 2 clusters
                score = silhouette_score(X, labels)
                results.append({'n_clusters': n_clusters, 'score': score})
                
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
                    best_labels = labels
        except Exception as e:
            print(f"Error with Hierarchical Clustering (n_clusters={n_clusters}): {str(e)}")
    
    return best_n_clusters, best_score, best_labels, results


def apply_dbscan(X, eps_values, min_samples_values):
    """Apply DBSCAN clustering with different eps and min_samples values and return the best result."""
    best_score = -1
    best_eps = None
    best_min_samples = None
    best_labels = None
    
    results = []
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)
                
                # Calculate silhouette score if more than one cluster is found
                unique_labels = np.unique(labels)
                
                # Check if we have at least 2 clusters (excluding noise points)
                valid_clusters = [label for label in unique_labels if label != -1]
                
                if len(valid_clusters) > 1:
                    # Filter out noise points for silhouette calculation
                    mask = labels != -1
                    if np.sum(mask) > len(valid_clusters):  # Ensure we have enough points
                        score = silhouette_score(X[mask], labels[mask])
                        results.append({'eps': eps, 'min_samples': min_samples, 'score': score})
                        
                        if score > best_score:
                            best_score = score
                            best_eps = eps
                            best_min_samples = min_samples
                            best_labels = labels
            except Exception as e:
                print(f"Error with DBSCAN (eps={eps}, min_samples={min_samples}): {str(e)}")
    
    return (best_eps, best_min_samples), best_score, best_labels, results


def analyze_dataset(dataset_name, results_dir='results'):
    """Analyze a dataset with different clustering algorithms and reduction methods."""
    print(f"\n{'='*50}\nAnalyzing dataset: {dataset_name}\n{'='*50}")
    
    # Create output directory for clustering results
    output_dir = os.path.join(results_dir, dataset_name, 'clustering')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load reduced data
    reduced_data = load_reduced_data(dataset_name, results_dir)
    if reduced_data is None:
        print(f"Error: Could not load reduced data for {dataset_name}")
        return None
    
    # Dictionary to store results
    clustering_results = {}
    
    # For each reduction method
    for method in reduction_methods:
        print(f"\nApplying clustering to {method.upper()} reduced data")
        
        # Get the reduced data for this method
        X = reduced_data[method]
        
        # Check if we have enough samples for clustering
        if X.shape[0] < 3:  # Need at least 3 samples for meaningful clustering
            print(f"  Not enough samples ({X.shape[0]}) for clustering. Skipping {method}.")
            continue
        
        # Dictionary to store results for this reduction method
        method_results = {}
        
        # Apply KMeans
        print("  Applying KMeans...")
        best_k, best_score, best_labels, all_results = apply_kmeans(X, kmeans_k_values)
        if best_k is not None:
            method_results['kmeans'] = {
                'optimal_params': {'k': best_k},
                'silhouette_score': best_score,
                'labels': best_labels,
                'all_results': all_results
            }
            print(f"    Optimal k: {best_k}, Silhouette Score: {best_score:.4f}")
        else:
            method_results['kmeans'] = {
                'optimal_params': {'k': None},
                'silhouette_score': None,
                'labels': None,
                'all_results': all_results
            }
            print("    No valid KMeans clustering found")
        
        # Apply Hierarchical Clustering
        print("  Applying Hierarchical Clustering...")
        best_n_clusters, best_score, best_labels, all_results = apply_hierarchical(X, hierarchical_n_clusters)
        if best_n_clusters is not None:
            method_results['hierarchical'] = {
                'optimal_params': {'n_clusters': best_n_clusters},
                'silhouette_score': best_score,
                'labels': best_labels,
                'all_results': all_results
            }
            print(f"    Optimal n_clusters: {best_n_clusters}, Silhouette Score: {best_score:.4f}")
        else:
            method_results['hierarchical'] = {
                'optimal_params': {'n_clusters': None},
                'silhouette_score': None,
                'labels': None,
                'all_results': all_results
            }
            print("    No valid Hierarchical clustering found")
        
        # Apply DBSCAN
        print("  Applying DBSCAN...")
        best_params, best_score, best_labels, all_results = apply_dbscan(X, dbscan_eps_values, dbscan_min_samples_values)
        if best_params[0] is not None:  # If a valid result was found
            method_results['dbscan'] = {
                'optimal_params': {'eps': best_params[0], 'min_samples': best_params[1]},
                'silhouette_score': best_score,
                'labels': best_labels,
                'all_results': all_results
            }
            print(f"    Optimal eps: {best_params[0]}, min_samples: {best_params[1]}, Silhouette Score: {best_score:.4f}")
        else:
            method_results['dbscan'] = {
                'optimal_params': {'eps': None, 'min_samples': None},
                'silhouette_score': None,
                'labels': None,
                'all_results': all_results
            }
            print("    No valid DBSCAN clustering found")
        
        # Store results for this reduction method
        clustering_results[method] = method_results
    
    # Save results to CSV
    save_results_to_csv(clustering_results, dataset_name, output_dir)
    
    # Create visualizations
    create_visualizations(clustering_results, reduced_data, dataset_name, output_dir)
    
    return clustering_results


def save_results_to_csv(clustering_results, dataset_name, output_dir):
    """Save clustering results to CSV files."""
    # Create a DataFrame for optimal parameters and scores
    optimal_results = []
    
    for reduction_method, method_results in clustering_results.items():
        for clustering_algorithm, results in method_results.items():
            row = {
                'Dataset': dataset_name,
                'Reduction_Method': reduction_method,
                'Clustering_Algorithm': clustering_algorithm,
                'Silhouette_Score': results['silhouette_score']
            }
            
            # Add optimal parameters
            for param_name, param_value in results['optimal_params'].items():
                row[f'Optimal_{param_name}'] = param_value
            
            optimal_results.append(row)
    
    if optimal_results:  # Only save if we have results
        # Create DataFrame and save to CSV
        optimal_df = pd.DataFrame(optimal_results)
        optimal_csv_path = os.path.join(output_dir, f'{dataset_name}_optimal_clustering.csv')
        optimal_df.to_csv(optimal_csv_path, index=False)
        print(f"Optimal clustering results saved to: {optimal_csv_path}")
    else:
        print(f"No valid clustering results found for {dataset_name}")
    
    # Save all trial results for each algorithm
    for reduction_method, method_results in clustering_results.items():
        for clustering_algorithm, results in method_results.items():
            if results['all_results']:  # Only save if we have results
                all_results_df = pd.DataFrame(results['all_results'])
                all_results_csv_path = os.path.join(
                    output_dir, 
                    f'{dataset_name}_{reduction_method}_{clustering_algorithm}_all_trials.csv'
                )
                all_results_df.to_csv(all_results_csv_path, index=False)
                print(f"All trials for {clustering_algorithm} on {reduction_method} saved to: {all_results_csv_path}")
            else:
                print(f"No valid trials found for {clustering_algorithm} on {reduction_method}")


def create_visualizations(clustering_results, reduced_data, dataset_name, output_dir):
    """Create visualizations of clustering results."""
    # For each reduction method
    for reduction_method, method_results in clustering_results.items():
        # Get the reduced data for this method
        X = reduced_data[reduction_method]
        
        # Check if we have at least 2D data for visualization
        if X.shape[1] < 2:
            print(f"Cannot create visualization for {reduction_method}: data has fewer than 2 dimensions")
            continue
        
        # Create a figure with subplots for each clustering algorithm
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Clustering Results for {dataset_name} - {reduction_method.upper()} Reduction', fontsize=16)
        
        # Plot KMeans results
        if 'kmeans' in method_results and method_results['kmeans']['labels'] is not None and method_results['kmeans']['silhouette_score'] is not None:
            axes[0].scatter(X[:, 0], X[:, 1], c=method_results['kmeans']['labels'], cmap='viridis', alpha=0.7)
            axes[0].set_title(f"KMeans (k={method_results['kmeans']['optimal_params']['k']})\nSilhouette Score: {method_results['kmeans']['silhouette_score']:.4f}")
        else:
            axes[0].text(0.5, 0.5, 'No valid KMeans clustering', ha='center', va='center')
            axes[0].set_title("KMeans")
        
        # Plot Hierarchical results
        if 'hierarchical' in method_results and method_results['hierarchical']['labels'] is not None and method_results['hierarchical']['silhouette_score'] is not None:
            axes[1].scatter(X[:, 0], X[:, 1], c=method_results['hierarchical']['labels'], cmap='viridis', alpha=0.7)
            axes[1].set_title(f"Hierarchical (n={method_results['hierarchical']['optimal_params']['n_clusters']})\nSilhouette Score: {method_results['hierarchical']['silhouette_score']:.4f}")
        else:
            axes[1].text(0.5, 0.5, 'No valid Hierarchical clustering', ha='center', va='center')
            axes[1].set_title("Hierarchical Clustering")
        
        # Plot DBSCAN results
        if 'dbscan' in method_results and method_results['dbscan']['labels'] is not None and method_results['dbscan']['silhouette_score'] is not None:
            axes[2].scatter(X[:, 0], X[:, 1], c=method_results['dbscan']['labels'], cmap='viridis', alpha=0.7)
            axes[2].set_title(f"DBSCAN (eps={method_results['dbscan']['optimal_params']['eps']}, min_samples={method_results['dbscan']['optimal_params']['min_samples']})\nSilhouette Score: {method_results['dbscan']['silhouette_score']:.4f}")
        else:
            axes[2].text(0.5, 0.5, 'No valid DBSCAN clustering', ha='center', va='center')
            axes[2].set_title("DBSCAN")
        
        # Set labels and grid for all subplots
        for ax in axes:
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_{reduction_method}_clustering.png'), dpi=300)
        plt.close()
        print(f"Visualization saved to: {os.path.join(output_dir, f'{dataset_name}_{reduction_method}_clustering.png')}")


def main():
    # Base directory for data files and results
    data_dir = 'd:\\data2'
    results_dir = os.path.join(data_dir, 'results')
    
    # Process each dataset
    all_results = {}
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        try:
            results = analyze_dataset(dataset, results_dir)
            if results is not None:
                all_results[dataset] = results
                print(f"Dataset {dataset} analyzed successfully!")
            else:
                print(f"Failed to analyze dataset {dataset}")
        except Exception as e:
            print(f"Error analyzing dataset {dataset}: {str(e)}")
    
    if all_results:
        print("\nAll datasets analyzed successfully!")
        print(f"Results saved to: {results_dir}")
    else:
        print("\nNo datasets were successfully analyzed.")
        print("Please check the error messages above.")
    
    return all_results


if __name__ == "__main__":
    main()