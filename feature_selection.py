import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.cluster import KMeans
import os
import argparse


def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset."""
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Check if there's an unnamed index column and drop it if present
    if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
        df = df.drop(df.columns[0], axis=1)
    
    # Identify categorical and numerical features
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
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


def apply_pca(X, feature_names, variance_threshold=0.9):
    """Apply PCA and return results."""
    print("\n===== PCA =====")
    
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    # Determine number of components for desired variance
    explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(explained_variance_ratio_cumsum >= variance_threshold) + 1
    
    print(f"Number of components explaining {variance_threshold*100}% variance: {n_components}")
    print(f"Explained variance by component: {pca.explained_variance_ratio_[:n_components]}")
    
    # Calculate total absolute loading across all components for each feature
    total_loadings = {}
    
    for i in range(n_components):
        # Get the absolute loadings for this component
        loadings = np.abs(pca.components_[i])
        
        # Add loadings to total for each feature
        for j, loading in enumerate(loadings):
            feature_name = feature_names[j] if j < len(feature_names) else f"Feature_{j}"
            if feature_name in total_loadings:
                total_loadings[feature_name] += loading
            else:
                total_loadings[feature_name] = loading
    
    # Sort features by total absolute loading
    sorted_features = sorted(total_loadings.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 5 features with highest total absolute loading
    top_5_features = sorted_features[:5]
    
    print("\n===== Top 5 Features by Total Absolute Loading Across All Components =====")
    for i, (feature, total_loading) in enumerate(top_5_features):
        print(f"  {i+1}. {feature}: {total_loading:.4f}")
    
    return X_pca[:, :n_components], n_components, top_5_features


def apply_lda(X, feature_names, target=None, n_clusters=3):
    """Apply LDA and return results."""
    print("\n===== LDA =====")
    
    # If no target is provided, generate pseudo-labels using KMeans
    if target is None:
        print("No class labels available. Generating pseudo-labels using KMeans.")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        target = kmeans.fit_predict(X)
    
    # Apply LDA
    n_components = min(len(np.unique(target)) - 1, X.shape[1])
    lda = LDA(n_components=n_components)
    X_lda = lda.fit_transform(X, target)
    
    print(f"Number of LDA components: {n_components}")
    
    # Get feature importance
    feature_importance = []
    all_features_importance = {}
    
    for i in range(n_components):
        # Get the absolute coefficients for this component
        coefficients = np.abs(lda.coef_[i] if lda.coef_.ndim > 1 else lda.coef_)
        # Get indices of top features
        top_indices = coefficients.argsort()[::-1]
        # Map indices to feature names
        top_features = [(feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}", coefficients[idx]) 
                        for idx in top_indices]
        feature_importance.append(top_features)
        
        # Aggregate feature importance across all components
        for feature, coef in top_features:
            if feature in all_features_importance:
                all_features_importance[feature] = max(all_features_importance[feature], coef)
            else:
                all_features_importance[feature] = coef
    
    # Print top features for each component
    for i, features in enumerate(feature_importance):
        print(f"\nLDA Component {i+1}")
        print("Top features with highest absolute coefficients:")
        for feature, coef in features[:5]:  # Show top 5 features
            print(f"  {feature}: {coef:.4f}")
    
    # Get top 5 features across all components
    top_features_overall = sorted(all_features_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("\n===== Top 5 Features Across All LDA Components =====")
    for feature, coef in top_features_overall:
        print(f"  {feature}: {coef:.4f}")
    
    return X_lda, feature_importance, top_features_overall


def apply_tsne(X):
    """Apply t-SNE and return results."""
    print("\n===== t-SNE =====")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    print("t-SNE reduction to 2 dimensions completed.")
    print("Note: t-SNE is primarily used for visualization and clustering, not feature interpretation.")
    
    return X_tsne


def apply_ifs(X, feature_names, target=None, method='f_test', k_values=[5, 10, 15]):
    """Apply Incremental Feature Selection and return results."""
    print("\n===== Incremental Feature Selection =====")
    
    # If no target is provided, generate pseudo-labels using KMeans
    if target is None:
        print("No class labels available. Generating pseudo-labels using KMeans for feature ranking.")
        kmeans = KMeans(n_clusters=3, random_state=42)
        target = kmeans.fit_predict(X)
    
    # Choose the scoring function
    if method == 'f_test':
        print("Using ANOVA F-test for feature ranking")
        score_func = f_classif
    else:  # method == 'mutual_info'
        print("Using Mutual Information for feature ranking")
        score_func = mutual_info_classif
    
    # Apply feature selection
    selector = SelectKBest(score_func=score_func, k='all')
    selector.fit(X, target)
    
    # Get feature scores and indices
    scores = selector.scores_
    if np.any(np.isnan(scores)):
        print("Warning: NaN scores detected. Replacing with zeros.")
        scores = np.nan_to_num(scores)
    
    # Sort features by importance
    indices = np.argsort(scores)[::-1]
    ranked_features = [(feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}", scores[idx]) 
                      for idx in indices]
    
    # Print top features for each k
    for k in k_values:
        k = min(k, len(ranked_features))
        print(f"\nTop {k} features:")
        for i, (feature, score) in enumerate(ranked_features[:k]):
            print(f"  {i+1}. {feature}: {score:.4f}")
    
    # Get top 5 features
    top_5_features = ranked_features[:5]
    print("\n===== Top 5 Features by IFS =====")
    for i, (feature, score) in enumerate(top_5_features):
        print(f"  {i+1}. {feature}: {score:.4f}")
    
    return ranked_features, top_5_features


def main():
    parser = argparse.ArgumentParser(description='Feature Selection and Dimensionality Reduction')
    parser.add_argument('--file', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--variance', type=float, default=0.9, help='Variance threshold for PCA (default: 0.9)')
    parser.add_argument('--clusters', type=int, default=3, help='Number of clusters for KMeans (default: 3)')
    parser.add_argument('--method', type=str, default='f_test', choices=['f_test', 'mutual_info'],
                        help='Feature ranking method for IFS (default: f_test)')
    args = parser.parse_args()
    
    # Get the dataset name from the file path
    dataset_name = os.path.basename(args.file).split('.')[0]
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Load and preprocess data
    X, target, feature_names, original_columns = load_and_preprocess_data(args.file)
    print(f"Dataset shape after preprocessing: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    if target is not None:
        print(f"Target variable available with {len(np.unique(target))} classes")
    
    # Apply PCA
    X_pca, n_components, pca_top_features = apply_pca(X, feature_names, args.variance)
    
    # Apply LDA
    X_lda, lda_features, lda_top_features = apply_lda(X, feature_names, target, args.clusters)
    
    # Apply t-SNE
    X_tsne = apply_tsne(X)
    
    # Apply IFS
    ifs_features, ifs_top_features = apply_ifs(X, feature_names, target, args.method)
    
    # Print summary of top features
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
    
    print("\nFeature selection and dimensionality reduction completed.")


if __name__ == "__main__":
    main()