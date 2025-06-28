import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer  # For handling missing values
from sklearn.compose import ColumnTransformer  # For preprocessing pipeline

# Import functions from feature_selection.py but we'll override load_and_preprocess_data
from feature_selection import apply_pca, apply_lda, apply_tsne, apply_ifs


def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset with handling for missing values."""
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Check if there's an unnamed index column and drop it if present
    if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
        df = df.drop(df.columns[0], axis=1)
    
    # Handle missing values
    # First, check the percentage of missing values in each column
    missing_percentage = df.isnull().mean() * 100
    
    # Drop columns with more than 50% missing values
    columns_to_drop = missing_percentage[missing_percentage > 50].index.tolist()
    if columns_to_drop:
        print(f"Dropping columns with >50% missing values: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    
    # For remaining columns with missing values, impute them
    # For numerical columns, use median imputation
    # For categorical columns, use most frequent value imputation
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Impute numerical features
    if numerical_features:
        num_imputer = SimpleImputer(strategy='median')
        df[numerical_features] = num_imputer.fit_transform(df[numerical_features])
    
    # Impute categorical features
    if categorical_features:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_features] = cat_imputer.fit_transform(df[categorical_features])
    
    # Remove target variable if it exists (assuming it's the last column)
    target = None
    if len(categorical_features) > 0 and df[categorical_features].nunique().max() < 10:
        # Use the categorical column with the fewest unique values as target
        target_candidates = [(col, df[col].nunique()) for col in categorical_features]
        target_candidates.sort(key=lambda x: x[1])
        if target_candidates:
            target_col = target_candidates[0][0]
            if target_col in numerical_features:
                numerical_features.remove(target_col)
            else:
                categorical_features.remove(target_col)
            target = df[target_col]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop'  # Drop columns not specified in transformers
    )
    
    # Apply preprocessing
    X = preprocessor.fit_transform(df)
    
    # Convert sparse matrix to dense if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Get feature names after preprocessing
    num_feature_names = numerical_features
    
    # Get one-hot encoded feature names
    if len(categorical_features) > 0:
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    else:
        cat_feature_names = []
    
    feature_names = num_feature_names + cat_feature_names
    
    return X, target, feature_names, df.columns.tolist()


def create_visualizations(X, X_pca, X_lda, X_tsne, target, dataset_name, output_dir):
    """Create and save visualizations for the dataset."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the figure for all visualizations
    plt.figure(figsize=(20, 15))
    
    # Convert target to numeric if it's categorical
    if target is not None and not np.issubdtype(target.dtype, np.number):
        # Create a mapping of unique values to integers
        unique_values = target.unique()
        target_map = {val: i for i, val in enumerate(unique_values)}
        target_numeric = np.array([target_map[val] for val in target])
        # Create a legend mapping for later use
        legend_map = {i: val for val, i in target_map.items()}
    else:
        target_numeric = target
        legend_map = None
    
    # 1. PCA Visualization
    plt.subplot(2, 2, 1)
    if target is not None:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=target_numeric, cmap='viridis', alpha=0.7)
        if legend_map:
            # Add a legend for categorical targets
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i/len(legend_map)), 
                                 markersize=10) for i in range(len(legend_map))]
            plt.legend(handles, legend_map.values(), title='Classes')
        else:
            plt.colorbar(scatter, label='Class')
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
    plt.title(f'PCA - {dataset_name}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. LDA Visualization (if more than 1 component)
    plt.subplot(2, 2, 2)
    if X_lda.shape[1] > 1:
        if target is not None:
            scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=target_numeric, cmap='viridis', alpha=0.7)
            if legend_map:
                # Add a legend for categorical targets
                handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i/len(legend_map)), 
                                     markersize=10) for i in range(len(legend_map))]
                plt.legend(handles, legend_map.values(), title='Classes')
            else:
                plt.colorbar(scatter, label='Class')
        else:
            plt.scatter(X_lda[:, 0], X_lda[:, 1], alpha=0.7)
        plt.title(f'LDA - {dataset_name}')
        plt.xlabel('LD 1')
        plt.ylabel('LD 2')
        plt.grid(True, linestyle='--', alpha=0.7)
    else:
        plt.text(0.5, 0.5, 'LDA has only one component', ha='center', va='center')
        plt.title(f'LDA - {dataset_name}')
        plt.axis('off')
    
    # 3. t-SNE Visualization
    plt.subplot(2, 2, 3)
    if target is not None:
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=target_numeric, cmap='viridis', alpha=0.7)
        if legend_map:
            # Add a legend for categorical targets
            handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.viridis(i/len(legend_map)), 
                                 markersize=10) for i in range(len(legend_map))]
            plt.legend(handles, legend_map.values(), title='Classes')
        else:
            plt.colorbar(scatter, label='Class')
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.7)
    plt.title(f't-SNE - {dataset_name}')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Feature Correlation Heatmap (using original data)
    plt.subplot(2, 2, 4)
    # Calculate correlation matrix for the first 10 features (to avoid overcrowding)
    if X.shape[1] > 10:
        corr_matrix = np.corrcoef(X[:, :10], rowvar=False)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title(f'Correlation Matrix (First 10 Features) - {dataset_name}')
    else:
        corr_matrix = np.corrcoef(X, rowvar=False)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title(f'Correlation Matrix - {dataset_name}')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_visualizations.png'), dpi=300)
    plt.close()
    
    # Create additional visualizations
    
    # 5. PCA Explained Variance
    plt.figure(figsize=(10, 6))
    pca = PCA()
    pca.fit(X)
    explained_variance = pca.explained_variance_ratio_
    
    # Plot cumulative explained variance
    plt.plot(np.cumsum(explained_variance), marker='o')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title(f'PCA Explained Variance - {dataset_name}')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_pca_variance.png'), dpi=300)
    plt.close()
    
    return


def save_feature_rankings(pca_top_features, lda_top_features, ifs_top_features, dataset_name, output_dir):
    """Save feature rankings to a CSV file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrames for each method
    pca_df = pd.DataFrame(pca_top_features, columns=['Feature', 'Total Loading'])
    pca_df['Rank'] = range(1, len(pca_df) + 1)
    pca_df['Method'] = 'PCA'
    
    lda_df = pd.DataFrame(lda_top_features, columns=['Feature', 'Coefficient'])
    lda_df['Rank'] = range(1, len(lda_df) + 1)
    lda_df['Method'] = 'LDA'
    
    ifs_df = pd.DataFrame(ifs_top_features, columns=['Feature', 'Score'])
    ifs_df['Rank'] = range(1, len(ifs_df) + 1)
    ifs_df['Method'] = 'IFS'
    
    # Combine all rankings
    all_rankings = pd.concat([
        pca_df[['Feature', 'Rank', 'Method', 'Total Loading']].rename(columns={'Total Loading': 'Score'}),
        lda_df[['Feature', 'Rank', 'Method', 'Coefficient']].rename(columns={'Coefficient': 'Score'}),
        ifs_df[['Feature', 'Rank', 'Method', 'Score']]
    ])
    
    # Save to CSV
    all_rankings.to_csv(os.path.join(output_dir, f'{dataset_name}_feature_rankings.csv'), index=False)
    
    return all_rankings


def process_dataset(file_path, output_base_dir='results', max_features=1000):
    """Process a single dataset and save results."""
    # Get dataset name from file path
    dataset_name = os.path.basename(file_path).split('.')[0]
    print(f"\n{'='*50}\nProcessing dataset: {dataset_name}\n{'='*50}")
    
    # Create output directory
    output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess data
    X, target, feature_names, original_columns = load_and_preprocess_data(file_path)
    print(f"Dataset shape after preprocessing: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    if target is not None:
        print(f"Target variable available with {len(np.unique(target))} classes")
    
    # Store original feature names before any reduction
    original_feature_names = feature_names.copy()
    
    # For very high-dimensional datasets, perform initial feature selection
    if X.shape[1] > max_features:
        print(f"Dataset has {X.shape[1]} features, reducing to {max_features} for analysis...")
        
        # Create a mapping to track original features
        feature_mapping = {}
        
        if target is not None:
            # If we have a target, use SelectKBest
            selector = SelectKBest(f_classif, k=max_features)
            X_reduced = selector.fit_transform(X, target)
            
            # Get selected feature indices and their scores
            selected_indices = selector.get_support(indices=True)
            scores = selector.scores_
            
            # Create mapping of original features to their importance scores
            for idx in selected_indices:
                if idx < len(original_feature_names):
                    feature_name = original_feature_names[idx]
                    score = scores[idx]
                    feature_mapping[feature_name] = score
            
            # Sort features by importance score
            sorted_features = sorted(feature_mapping.items(), key=lambda x: x[1], reverse=True)
            
            # Update feature names with original names
            feature_names = [name for name, _ in sorted_features]
            
            print(f"Top 5 original features selected by F-test:")
            for i, (feature, score) in enumerate(sorted_features[:5]):
                print(f"  {i+1}. {feature}: {score:.4f}")
        else:
            # If no target, use PCA for initial reduction but preserve original feature importance
            initial_pca = PCA(n_components=max_features)
            X_reduced = initial_pca.fit_transform(X)
            
            # Calculate feature importance based on PCA loadings
            for i, feature_name in enumerate(original_feature_names):
                if i < len(initial_pca.components_[0]):
                    # Sum of absolute loadings across all components
                    importance = np.sum(np.abs(initial_pca.components_[:, i]))
                    feature_mapping[feature_name] = importance
            
            # Sort features by importance
            sorted_features = sorted(feature_mapping.items(), key=lambda x: x[1], reverse=True)
            
            # Take top max_features features
            top_features = sorted_features[:max_features]
            
            # Update feature names with original names
            feature_names = [name for name, _ in top_features]
            
            print(f"Top 5 original features preserved in PCA reduction:")
            for i, (feature, importance) in enumerate(top_features[:5]):
                print(f"  {i+1}. {feature}: {importance:.4f}")
        
        print(f"Reduced dataset shape: {X_reduced.shape}")
        X = X_reduced
    
    # Apply PCA with original feature names
    X_pca, n_components, pca_top_features = apply_pca(X, feature_names)
    
    # Apply LDA with original feature names
    X_lda, lda_features, lda_top_features = apply_lda(X, feature_names, target)
    
    # Apply t-SNE
    X_tsne = apply_tsne(X)
    
    # Apply IFS with original feature names
    ifs_features, ifs_top_features = apply_ifs(X, feature_names, target)
    
    # Create and save visualizations
    create_visualizations(X, X_pca, X_lda, X_tsne, target, dataset_name, output_dir)
    
    # Save feature rankings
    all_rankings = save_feature_rankings(pca_top_features, lda_top_features, ifs_top_features, dataset_name, output_dir)
    
    # Print summary
    print("\n===== SUMMARY OF TOP FEATURES =====")
    
    print("\nPCA: Top 5 features with highest total absolute loading across all components")
    for i, (feature, total_loading) in enumerate(pca_top_features):
        print(f"  {i+1}. {feature}: {total_loading:.4f}")
    
    print("\nLDA: Top 5 features with highest absolute coefficient values")
    for feature, coef in lda_top_features:
        print(f"  {feature}: {coef:.4f}")
    
    print("\nIFS: Top 5 features ranked by importance")
    for i, (feature, score) in enumerate(ifs_top_features):
        print(f"  {i+1}. {feature}: {score:.4f}")
    
    print("\nt-SNE: No feature interpretation (used for visualization only)")
    
    print(f"\nResults saved to: {output_dir}")
    
    return output_dir


def main():
    # Base directory for data files
    data_dir = 'd:\\data2'
    
    # List of datasets to process
    datasets = [
        '1_CustomerPersonality.csv',
        '2_SupermarketAnalysis.csv',
        '3_CreditCardFraud.csv',
        '4_MedicalCost.csv'
    ]
    
    # Create results directory
    results_dir = os.path.join(data_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Process each dataset
    for dataset in datasets:
        file_path = os.path.join(data_dir, dataset)
        process_dataset(file_path, results_dir)
    
    print("\nAll datasets processed successfully!")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()