"""
SIMPLE MACHINE LEARNING BASELINES FOR DEFORESTATION DETECTION
============================================================

This module implements traditional machine learning algorithms as baselines
for deforestation detection. It extracts statistical features from satellite
imagery and trains simple models for comparison with deep learning approaches.

Algorithms implemented:
- Random Forest: Ensemble of decision trees
- Logistic Regression: Linear classifier with sigmoid activation
- K-Nearest Neighbors (KNN): Instance-based learning

Author: AI Assistant
Date: 2024
"""

import numpy as np
import rasterio  # For reading satellite imagery
import os
import pickle  # For saving trained models
import json    # For saving results
import config

def extract_features_from_patch(patch_idx, use_sentinel1=True, use_sentinel2=True):
    """
    Extract statistical features from satellite data patches
    
    This function converts high-dimensional satellite imagery into a compact
    feature vector by computing statistical measures for each band. This is
    necessary because traditional ML algorithms can't handle raw image data.
    
    Features extracted per band:
    - Mean: Average pixel value
    - Standard deviation: Variability in pixel values
    - Median: Middle value (robust to outliers)
    - 25th/75th percentiles: Quartiles for distribution shape
    - Min/Max: Range of values
    - Variance: Square of standard deviation
    - Range: Difference between max and min
    
    Args:
        patch_idx: Index of the satellite patch to process
        use_sentinel1: Whether to include Sentinel-1 radar features
        use_sentinel2: Whether to include Sentinel-2 optical features
    
    Returns:
        np.array: Feature vector of statistical measures
    """
    features = []
    
    # Extract features from Sentinel-1 radar data
    if use_sentinel1:
        sentinel1_path = f"{config.SENTINEL1_PATH}/RASTER_{patch_idx}.tif"
        with rasterio.open(sentinel1_path) as src:
            s1_data = src.read()  # Shape: (bands, height, width)
            
            # Process each radar band (VV, VH polarizations)
            for band in range(s1_data.shape[0]):
                band_data = s1_data[band].flatten()  # Flatten to 1D array
                
                # Extract comprehensive statistical features
                features.extend([
                    np.mean(band_data),           # Average intensity
                    np.std(band_data),            # Standard deviation
                    np.median(band_data),         # Median (robust to outliers)
                    np.percentile(band_data, 25), # First quartile
                    np.percentile(band_data, 75), # Third quartile
                    np.min(band_data),            # Minimum value
                    np.max(band_data),            # Maximum value
                    np.var(band_data),            # Variance
                    np.ptp(band_data),            # Range (max - min)
                ])
    
    # Extract features from Sentinel-2 optical data
    if use_sentinel2:
        sentinel2_path = f"{config.SENTINEL2_PATH}/RASTER_{patch_idx}.tif"
        with rasterio.open(sentinel2_path) as src:
            s2_data = src.read()  # Shape: (bands, height, width)
            
            # Process each optical band (Red, Green, Blue, NIR)
            for band in range(s2_data.shape[0]):
                band_data = s2_data[band].flatten()  # Flatten to 1D array
                
                # Extract comprehensive statistical features
                features.extend([
                    np.mean(band_data),           # Average intensity
                    np.std(band_data),            # Standard deviation
                    np.median(band_data),         # Median (robust to outliers)
                    np.percentile(band_data, 25), # First quartile
                    np.percentile(band_data, 75), # Third quartile
                    np.min(band_data),            # Minimum value
                    np.max(band_data),            # Maximum value
                    np.var(band_data),            # Variance
                    np.ptp(band_data),            # Range (max - min)
                ])
    
    return np.array(features)

def load_data_for_ml(use_sentinel1=True, use_sentinel2=True):
    """Load and prepare data for machine learning models"""
    print(f"Loading data - Sentinel-1: {use_sentinel1}, Sentinel-2: {use_sentinel2}")
    
    X = []
    y = []
    
    for patch_idx in range(config.NUM_PATCHES):
        features = extract_features_from_patch(patch_idx, use_sentinel1, use_sentinel2)
        X.append(features)
        
        # Load deforestation mask
        mask_path = f"{config.MASK_PATH}/RASTER_{patch_idx}.tif"
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            mask_binary = (mask == 1).astype(int)
            # Calculate deforestation rate
            deforestation_rate = np.mean(mask_binary)
            # Convert to binary: >10% deforestation = positive
            y.append(1 if deforestation_rate > 0.1 else 0)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    print(f"Positive samples: {np.sum(y)}, Negative samples: {np.sum(1-y)}")
    
    return X, y

class SimpleRandomForest:
    """A simple implementation of Random Forest using decision stumps"""
    
    def __init__(self, n_trees=50, max_depth=3):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
    
    def _create_decision_stump(self, X, y):
        """Create a simple decision stump"""
        best_feature = 0
        best_threshold = 0
        best_gini = float('inf')
        
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            # Try different thresholds for this feature
            values = np.unique(X[:, feature])
            for threshold in values[1:]:  # Skip first value
                # Split data
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate gini impurity
                left_gini = self._gini_impurity(y[left_mask])
                right_gini = self._gini_impurity(y[right_mask])
                
                weighted_gini = (np.sum(left_mask) * left_gini + np.sum(right_mask) * right_gini) / n_samples
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left_pred': np.mean(y[X[:, best_feature] <= best_threshold]) > 0.5,
            'right_pred': np.mean(y[X[:, best_feature] > best_threshold]) > 0.5
        }
    
    def _gini_impurity(self, y):
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        p1 = np.mean(y)
        p0 = 1 - p1
        return 1 - p1**2 - p0**2
    
    def fit(self, X, y):
        """Train the random forest"""
        self.trees = []
        n_samples, n_features = X.shape
        
        for i in range(self.n_trees):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Create decision stump
            tree = self._create_decision_stump(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
    
    def predict(self, X):
        """Make predictions"""
        predictions = []
        for x in X:
            votes = []
            for tree in self.trees:
                if x[tree['feature']] <= tree['threshold']:
                    votes.append(tree['left_pred'])
                else:
                    votes.append(tree['right_pred'])
            
            # Majority vote
            predictions.append(int(np.mean(votes) > 0.5))
        
        return np.array(predictions)

class SimpleLogisticRegression:
    """A simple implementation of Logistic Regression"""
    
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Clip to prevent overflow
    
    def fit(self, X, y):
        """Train logistic regression using gradient descent"""
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iter):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(z)
            
            # Calculate loss
            loss = -np.mean(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
            
            # Calculate gradients
            dw = np.dot(X.T, (predictions - y)) / n_samples
            db = np.mean(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """Make predictions"""
        z = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(z)
        return (predictions > 0.5).astype(int)

class SimpleKNN:
    """A simple implementation of K-Nearest Neighbors"""
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store training data"""
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """Make predictions"""
        predictions = []
        for x in X:
            # Calculate distances
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            
            # Find k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            
            # Majority vote
            predictions.append(int(np.mean(k_labels) > 0.5))
        
        return np.array(predictions)

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics"""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1

def train_ml_model(model, model_name, X_train, y_train, X_val, y_val):
    """Train a machine learning model and return results"""
    print(f"\nTraining {model_name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Calculate metrics
    train_acc, train_prec, train_rec, train_f1 = calculate_metrics(y_train, y_pred_train)
    val_acc, val_prec, val_rec, val_f1 = calculate_metrics(y_val, y_pred_val)
    
    train_metrics = {
        'accuracy': train_acc,
        'precision': train_prec,
        'recall': train_rec,
        'f1': train_f1
    }
    
    val_metrics = {
        'accuracy': val_acc,
        'precision': val_prec,
        'recall': val_rec,
        'f1': val_f1
    }
    
    print(f"Training - Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}")
    print(f"Validation - Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}")
    
    return model, train_metrics, val_metrics

def save_ml_model(model, model_name, train_metrics, val_metrics):
    """Save machine learning model and results"""
    os.makedirs('trained_models', exist_ok=True)
    
    # Save model
    model_path = f'trained_models/{model_name.lower()}_simple_ml_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save results
    results_path = f'trained_models/{model_name.lower()}_simple_ml_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'model_name': model_name,
            'model_type': type(model).__name__,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        }, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Results saved to: {results_path}")

def create_stratified_split(X, y, test_size=0.3):
    """Create stratified train/validation split"""
    # Separate positive and negative samples
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    # Shuffle
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)
    
    # Split each class
    pos_train_size = max(1, int(len(pos_indices) * (1 - test_size)))
    neg_train_size = max(1, int(len(neg_indices) * (1 - test_size)))
    
    pos_train = pos_indices[:pos_train_size]
    pos_val = pos_indices[pos_train_size:]
    neg_train = neg_indices[:neg_train_size]
    neg_val = neg_indices[neg_train_size:]
    
    # Combine
    train_indices = np.concatenate([pos_train, neg_train])
    val_indices = np.concatenate([pos_val, neg_val])
    
    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]

def main():
    print("=" * 60)
    print("SIMPLE MACHINE LEARNING BASELINES FOR DEFORESTATION DETECTION")
    print("=" * 60)
    
    # Set random seed
    np.random.seed(42)
    
    # Define different data configurations
    data_configs = [
        ("Sentinel1_Only", True, False),
        ("Sentinel2_Only", False, True),
        ("Combined_Data", True, True)
    ]
    
    # Define simple ML models
    ml_models = [
        ("RandomForest", SimpleRandomForest(n_trees=50, max_depth=3)),
        ("LogisticRegression", SimpleLogisticRegression(learning_rate=0.01, max_iter=1000)),
        ("KNN", SimpleKNN(k=3))
    ]
    
    all_results = []
    
    for data_name, use_s1, use_s2 in data_configs:
        print(f"\n{'='*60}")
        print(f"DATA CONFIGURATION: {data_name}")
        print(f"{'='*60}")
        
        # Load data
        X, y = load_data_for_ml(use_sentinel1=use_s1, use_sentinel2=use_s2)
        
        # Create stratified train/validation split
        X_train, X_val, y_train, y_val = create_stratified_split(X, y, test_size=0.3)
        
        print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Val set: {X_val.shape[0]} samples")
        
        for model_name, model in ml_models:
            full_model_name = f"{model_name}_{data_name}"
            print(f"\n{'-'*40}")
            print(f"Training {full_model_name}")
            print(f"{'-'*40}")
            
            try:
                # Train model
                trained_model, train_metrics, val_metrics = train_ml_model(
                    model, full_model_name, X_train, y_train, X_val, y_val
                )
                
                # Save model
                save_ml_model(trained_model, full_model_name, train_metrics, val_metrics)
                
                # Store results
                all_results.append({
                    'model_name': full_model_name,
                    'data_config': data_name,
                    'ml_algorithm': model_name,
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall']
                })
                
            except Exception as e:
                print(f"Error training {full_model_name}: {e}")
                continue
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SIMPLE ML BASELINES SUMMARY")
    print(f"{'='*60}")
    
    # Sort by validation accuracy
    all_results_sorted = sorted(all_results, key=lambda x: x['val_accuracy'], reverse=True)
    
    print(f"{'Rank':<4} {'Model':<25} {'Data':<15} {'Algorithm':<15} {'Accuracy':<10} {'F1':<10}")
    print("-" * 85)
    
    for i, result in enumerate(all_results_sorted):
        rank = i + 1
        print(f"{rank:<4} {result['model_name']:<25} {result['data_config']:<15} {result['ml_algorithm']:<15} "
              f"{result['val_accuracy']:<10.4f} {result['val_f1']:<10.4f}")
    
    # Save summary results
    summary_path = 'trained_models/simple_ml_baselines_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
