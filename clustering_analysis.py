import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Function to detect and remove outliers using IQR method
def remove_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out outliers
    filtered_df = df.copy()
    for column in df.columns:
        filtered_df = filtered_df[filtered_df[column] <= upper_bound[column]]
        filtered_df = filtered_df[filtered_df[column] >= lower_bound[column]]
    
    return filtered_df

# Function to detect and remove outliers using Isolation Forest
def remove_outliers_isolation_forest(df):
    clf = IsolationForest(random_state=42, contamination=0.05)
    outlier_pred = clf.fit_predict(df)
    return df[outlier_pred == 1]

# Function to preprocess data
def preprocess_data(df):
    # Handle missing values
    df = df.copy()
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Handle missing values in numeric columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Convert categorical variables to one-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_encoded)
    
    # Apply outlier detection and removal (using Isolation Forest)
    scaled_df = pd.DataFrame(scaled_data, columns=df_encoded.columns)
    cleaned_df = remove_outliers_isolation_forest(scaled_df)
    
    # Apply PCA if more than 10 features
    if df_encoded.shape[1] > 10:
        pca = PCA(n_components=0.95, random_state=42)  # Retain 95% variance
        pca_data = pca.fit_transform(cleaned_df)
        print(f"Reduced from {df_encoded.shape[1]} to {pca_data.shape[1]} features with PCA")
        return pca_data
    
    return cleaned_df.values

# Function to evaluate clustering with silhouette score
def evaluate_clustering(data, labels):
    # Check if there are at least 2 clusters
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return -1  # Invalid clustering
    
    # Check if any cluster has fewer than 5 points
    for label in unique_labels:
        if label != -1 and np.sum(labels == label) < 5:
            return -1  # Invalid clustering
    
    # Calculate silhouette score (ignoring noise points with label -1)
    if -1 in unique_labels:
        # Remove noise points for silhouette calculation
        valid_indices = labels != -1
        if np.sum(valid_indices) < 2 or len(np.unique(labels[valid_indices])) < 2:
            return -1  # Not enough valid points or clusters
        return silhouette_score(data[valid_indices], labels[valid_indices])
    else:
        return silhouette_score(data, labels)

# Function to run KMeans clustering
def run_kmeans(data):
    best_score = -1
    best_params = {}
    
    # Try more cluster values
    for k in range(2, 16):
        print(f"  - Testing KMeans with k={k}")
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
        labels = kmeans.fit_predict(data)
        score = evaluate_clustering(data, labels)
        
        print(f"    Silhouette score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_params = {'n_clusters': k}
    
    return best_score, best_params

# Function to run Agglomerative clustering
def run_agglomerative(data):
    best_score = -1
    best_params = {}
    
    # Try more cluster values
    for n_clusters in range(2, 16):
        print(f"  - Testing Agglomerative with n_clusters={n_clusters}")
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = agg.fit_predict(data)
        score = evaluate_clustering(data, labels)
        
        print(f"    Silhouette score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_params = {'n_clusters': n_clusters}
    
    return best_score, best_params

# Function to run DBSCAN clustering
def run_dbscan(data):
    best_score = -1
    best_params = {}
    
    # Expanded parameter grid for DBSCAN
    eps_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    min_samples_values = [3, 4, 5, 6, 7, 8, 10, 15]
    
    total_combinations = len(eps_values) * len(min_samples_values)
    current_combination = 0
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            current_combination += 1
            if current_combination % 10 == 0:
                print(f"  - Testing DBSCAN combination {current_combination}/{total_combinations}")
                
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)
            
            # Skip if all points are noise
            if len(np.unique(labels)) <= 1:
                continue
                
            score = evaluate_clustering(data, labels)
            
            if score > best_score:
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                print(f"    New best: eps={eps}, min_samples={min_samples}, clusters={n_clusters}, score={score:.4f}")
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}
    
    return best_score, best_params

# Main function to process all datasets
def process_datasets():
    # List of datasets
    datasets = [
        'CreditCardFraud_C1.csv',
        'CreditCardFraud_C2.csv',
        'CustomerPersonality_D1.csv',
        'CustomerPersonality_D2.csv',
        'CustomerPersonality_D3.csv',
        'MedicalCost_E1.csv',
        'SupermarketAnalysis_D1.csv',
        'SupermarketAnalysis_D2.csv'
    ]
    
    # Results storage
    results = []
    
    for i, dataset_name in enumerate(datasets):
        print(f"\nProcessing dataset {i+1}/{len(datasets)}: {dataset_name}")
        
        # Load dataset
        df = pd.read_csv(dataset_name)
        print(f"Dataset shape: {df.shape}")
        
        # Preprocess data
        processed_data = preprocess_data(df)
        print(f"Processed data shape: {processed_data.shape}")
        
        # Run clustering algorithms
        print("Running KMeans...")
        kmeans_score, kmeans_params = run_kmeans(processed_data)
        
        print("Running DBSCAN...")
        dbscan_score, dbscan_params = run_dbscan(processed_data)
        
        print("Running Agglomerative Clustering...")
        agg_score, agg_params = run_agglomerative(processed_data)
        
        # Store results
        if kmeans_score > -1:
            results.append({
                'Dataset Name': dataset_name,
                'Algorithm': 'KMeans',
                'Parameters Used': str(kmeans_params),
                '# Clusters': kmeans_params.get('n_clusters', 0),
                'Silhouette Score': kmeans_score
            })
        
        if dbscan_score > -1:
            # Get number of clusters (excluding noise)
            dbscan = DBSCAN(eps=dbscan_params['eps'], min_samples=dbscan_params['min_samples'])
            labels = dbscan.fit_predict(processed_data)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            results.append({
                'Dataset Name': dataset_name,
                'Algorithm': 'DBSCAN',
                'Parameters Used': str(dbscan_params),
                '# Clusters': n_clusters,
                'Silhouette Score': dbscan_score
            })
        
        if agg_score > -1:
            results.append({
                'Dataset Name': dataset_name,
                'Algorithm': 'Agglomerative',
                'Parameters Used': str(agg_params),
                '# Clusters': agg_params.get('n_clusters', 0),
                'Silhouette Score': agg_score
            })
    
    # Create and save results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('clustering_silhouette_scores_summary.csv', index=False)
    print("\nResults saved to clustering_silhouette_scores_summary.csv")
    
    return results_df

# Run the analysis
if __name__ == "__main__":
    results = process_datasets()
    
    # Display top results
    print("\nTop clustering results by silhouette score:")
    print(results.sort_values('Silhouette Score', ascending=False).head(10))