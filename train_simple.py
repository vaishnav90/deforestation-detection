"""
DEFORESTATION DETECTION MODEL TRAINING SCRIPT
============================================

This script trains deep learning models to detect deforestation in satellite imagery.
It supports multiple model architectures and data sources (Sentinel-1 radar, Sentinel-2 optical).

Key Features:
- Multiple CNN architectures (SimpleCNN, RobustBaselineCNN, LateFusionCNN)
- Support for Sentinel-1 (radar) and Sentinel-2 (optical) data
- Robust training with early stopping, gradient clipping, and focal loss
- Model checkpointing and result visualization
- Stratified train/validation splits to handle class imbalance

Author: AI Assistant
Date: 2024
"""

# Core PyTorch libraries for deep learning
import torch                    # Main PyTorch library for tensors and neural networks
import torch.nn as nn           # Neural network modules (layers, activations, etc.)
import torch.nn.functional as F # Functional operations (losses, activations, etc.)
import torch.optim as optim     # Optimization algorithms (Adam, SGD, etc.)
from torch.utils.data import DataLoader  # Data loading utilities

# Scientific computing and visualization
import numpy as np              # Numerical computing
from tqdm import tqdm          # Progress bars for training loops
import matplotlib.pyplot as plt # Plotting and visualization

# System and file operations
import os                       # File system operations
import torchvision.transforms as transforms  # Image augmentation transforms
import rasterio                 # Geospatial raster data reading
import pickle                   # Python object serialization
import json                     # JSON file handling

# Project-specific imports
import config                   # Configuration parameters (batch size, learning rate, etc.)
from data_loader import DeforestationDataset  # Custom dataset class for satellite data
from models import SimpleCNN, RobustBaselineCNN, LateFusionCNN  # Model architectures

def calculate_metrics(predictions, targets, threshold=0.5, debug=False):
    """
    Calculate classification metrics for binary deforestation detection
    
    This function converts continuous deforestation predictions to binary classifications
    and computes standard evaluation metrics (accuracy, precision, recall, F1).
    
    IMPORTANT: Uses fixed thresholds to avoid data leakage - no threshold optimization
    on validation set, which would be cheating and give overly optimistic results.
    
    Args:
        predictions: Model predictions (0-1 range from sigmoid activation)
        targets: Ground truth deforestation rates (0-1 range, continuous)
        threshold: Fixed threshold for binary classification (0.5 is realistic)
        debug: Whether to print debug information for troubleshooting
    
    Returns:
        tuple: (accuracy, precision, recall, f1_score)
    """
    # Convert model predictions to binary using realistic fixed threshold
    # 0.5 threshold means: if model predicts >50% deforestation probability, classify as "deforested"
    pred_binary = (predictions > threshold).float()
    
    # Convert continuous deforestation rates to binary labels
    # Use realistic threshold: >10% deforestation = positive class (deforested)
    # This means: if >10% of pixels in a patch are deforested, label the patch as "deforested"
    targets_binary = (targets > 0.1).float()
    
    # Flatten tensors to 1D arrays for easier computation
    pred_flat = pred_binary.cpu().numpy().flatten()
    target_flat = targets_binary.cpu().numpy().flatten()
    
    # Calculate confusion matrix components
    # TP = True Positive: correctly predicted deforestation
    # FP = False Positive: incorrectly predicted deforestation (false alarm)
    # FN = False Negative: missed deforestation (missed detection)
    # TN = True Negative: correctly predicted no deforestation
    tp = np.sum((pred_flat == 1) & (target_flat == 1))  # True Positives
    fp = np.sum((pred_flat == 1) & (target_flat == 0))  # False Positives
    fn = np.sum((pred_flat == 0) & (target_flat == 1))  # False Negatives
    tn = np.sum((pred_flat == 0) & (target_flat == 0))  # True Negatives
    
    # Debug output for troubleshooting (only shown for first few epochs)
    if debug:
        print(f"         Debug - Threshold: {threshold:.4f}")
        print(f"         Debug - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        print(f"         Debug - Pred binary: {pred_flat}")
        print(f"         Debug - Target binary: {target_flat}")
        print(f"         Debug - Raw predictions: {predictions.cpu().numpy().flatten()}")
        print(f"         Debug - Raw targets: {targets.cpu().numpy().flatten()}")
    
    # Calculate metrics with proper error handling to avoid division by zero
    total_samples = tp + fp + fn + tn
    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Standard classification metrics
    accuracy = (tp + tn) / total_samples                    # Overall correctness
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Of predicted positive, how many were correct?
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0      # Of actual positive, how many did we catch?
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0  # Harmonic mean of precision and recall
    
    return accuracy, precision, recall, f1


def analyze_class_distribution(dataset, model_name):
    """
    Analyze class distribution in the dataset to understand data characteristics
    
    This function examines the deforestation rates across all samples in the dataset
    to understand class imbalance and data distribution. This is crucial for:
    - Choosing appropriate loss functions (e.g., focal loss for imbalanced data)
    - Understanding if we have enough samples of each class
    - Detecting potential issues with the dataset
    
    Args:
        dataset: PyTorch dataset containing satellite imagery and masks
        model_name: Name of the model for reporting purposes
    
    Returns:
        float: Ratio of positive samples (deforestation > 10%)
    """
    # Calculate deforestation rate for each sample in the dataset
    deforestation_rates = []
    for i in range(len(dataset)):
        _, mask = dataset[i]  # Get the deforestation mask
        deforestation_rate = torch.mean(mask).item()  # Calculate mean deforestation rate
        deforestation_rates.append(deforestation_rate)
    
    deforestation_rates = np.array(deforestation_rates)
    
    # Count samples by class using the same threshold as in calculate_metrics
    positive_samples = np.sum(deforestation_rates > 0.1)  # >10% deforestation = positive
    total_samples = len(deforestation_rates)
    
    # Print detailed analysis
    print(f"\n{model_name} Dataset Analysis:")
    print(f"Total samples: {total_samples}")
    print(f"Positive samples (deforestation > 10%): {positive_samples}")
    print(f"Negative samples: {total_samples - positive_samples}")
    print(f"Class ratio (pos:neg): {positive_samples}:{total_samples - positive_samples}")
    print(f"Mean deforestation rate: {np.mean(deforestation_rates):.4f}")
    print(f"Std deforestation rate: {np.std(deforestation_rates):.4f}")
    
    # Return positive class ratio for use in loss function selection
    return positive_samples / total_samples if total_samples > 0 else 0

def save_model_checkpoint(model, model_name, results, train_indices, val_indices, epoch):
    """
    Save model checkpoint with comprehensive training metadata
    
    This function saves the trained model in multiple formats for different use cases:
    1. Model weights (.pth) - for loading the trained model
    2. Full checkpoint (.pkl) - for resuming training or detailed analysis
    3. Results summary (.json) - for easy reading and comparison
    
    Args:
        model: Trained PyTorch model
        model_name: Name identifier for the model
        results: Dictionary containing training metrics and history
        train_indices: Indices of samples used for training
        val_indices: Indices of samples used for validation
        epoch: Epoch number when model was saved
    """
    # Create comprehensive checkpoint with all training information
    checkpoint = {
        'model_state_dict': model.state_dict(),  # Model weights
        'model_name': model_name,                # Model identifier
        'results': results,                      # Training metrics history
        'train_indices': train_indices,          # Training data indices
        'val_indices': val_indices,             # Validation data indices
        'epoch': epoch,                         # Epoch number
        'config': {                             # Training configuration
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'num_epochs': config.NUM_EPOCHS,
            'device': config.DEVICE,
            'sentinel1_bands': config.SENTINEL1_BANDS,
            'sentinel2_bands': config.SENTINEL2_BANDS
        }
    }
    
    # Create models directory if it doesn't exist
    os.makedirs('trained_models', exist_ok=True)
    
    # Save model state dict (just the weights) - most common format for inference
    model_path = f'trained_models/{model_name.lower()}_best.pth'
    torch.save(model.state_dict(), model_path)
    
    # Save full checkpoint with metadata - useful for resuming training
    checkpoint_path = f'trained_models/{model_name.lower()}_checkpoint.pkl'
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    # Save results as JSON for easy reading and comparison between models
    results_path = f'trained_models/{model_name.lower()}_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'model_name': model_name,
            'best_f1': results['best_f1'],                    # Best F1 score achieved
            'best_epoch': results['best_epoch'],              # Epoch with best performance
            'final_train_loss': results['train_losses'][-1] if results['train_losses'] else 0,
            'final_val_loss': results['val_losses'][-1] if results['val_losses'] else 0,
            'final_train_f1': results['train_f1s'][-1] if results['train_f1s'] else 0,
            'final_val_f1': results['val_f1s'][-1] if results['val_f1s'] else 0,
            'total_epochs_trained': len(results['train_losses']),
            'train_indices': train_indices,                    # For reproducibility
            'val_indices': val_indices                         # For reproducibility
        }, f, indent=2)
    
    print(f"Model saved to: {model_path}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Results saved to: {results_path}")

def load_model_checkpoint(model_name, model_class, input_channels):
    """Load a previously trained model"""
    checkpoint_path = f'trained_models/{model_name.lower()}_checkpoint.pkl'
    
    if not os.path.exists(checkpoint_path):
        return None, None
    
    print(f"Loading model from: {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Initialize model
    if model_class == "LateFusion":
        model = LateFusionCNN()
    else:
        model = model_class(input_channels)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def check_if_model_exists(model_name):
    """Check if a trained model already exists"""
    checkpoint_path = f'trained_models/{model_name.lower()}_checkpoint.pkl'
    model_path = f'trained_models/{model_name.lower()}_best.pth'
    
    return os.path.exists(checkpoint_path) and os.path.exists(model_path)

def train_model(model_name, model_class, input_channels, use_sentinel1, use_sentinel2):
    """
    Main training function for deforestation detection models
    
    This function handles the complete training pipeline including:
    - Data preparation and stratified splitting
    - Model initialization and configuration
    - Training loop with validation
    - Early stopping and model checkpointing
    - Comprehensive logging and monitoring
    
    Args:
        model_name: String identifier for the model
        model_class: PyTorch model class (SimpleCNN, RobustBaselineCNN, etc.)
        input_channels: Number of input channels (depends on data sources)
        use_sentinel1: Whether to use Sentinel-1 radar data
        use_sentinel2: Whether to use Sentinel-2 optical data
    
    Returns:
        dict: Training results including metrics history and best performance
    """
    # Set up device (CPU, GPU, or Apple Silicon)
    device = torch.device(config.DEVICE)
    print(f"Training {model_name} on {device}")
    
    # Set random seeds for reproducibility - crucial for consistent results
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create stratified train/validation split to ensure balanced representation
    def create_stratified_split():
        """
        Create stratified train/validation split to handle class imbalance
        
        This function ensures that both training and validation sets contain
        samples from both classes (deforested and non-deforested). Without
        stratification, we might end up with all positive samples in validation
        and all negative samples in training, leading to unreliable evaluation.
        """
        # First, analyze deforestation rates for all patches to understand class distribution
        deforestation_rates = []
        for patch_idx in range(config.NUM_PATCHES):
            mask_path = f"{config.MASK_PATH}/RASTER_{patch_idx}.tif"
            with rasterio.open(mask_path) as src:
                mask = src.read(1)  # Read the deforestation mask
                mask_binary = (mask == 1).astype(np.float32)  # Convert to binary (1=deforested, 0=not)
                deforestation_rate = np.mean(mask_binary)  # Calculate deforestation percentage
                deforestation_rates.append(deforestation_rate)
        
        # Split patches into positive (deforested) and negative (non-deforested) classes
        # Using same threshold as in calculate_metrics: >10% deforestation = positive
        positive_patches = [i for i, rate in enumerate(deforestation_rates) if rate > 0.1]
        negative_patches = [i for i, rate in enumerate(deforestation_rates) if rate <= 0.1]
        
        # Shuffle both classes to ensure random selection
        np.random.shuffle(positive_patches)
        np.random.shuffle(negative_patches)
        
        # Split positive patches between train and validation
        pos_train_size = max(1, int(len(positive_patches) * config.TRAIN_SPLIT))
        pos_train = positive_patches[:pos_train_size]
        pos_val = positive_patches[pos_train_size:]
        
        # Split negative patches between train and validation
        neg_train_size = max(1, int(len(negative_patches) * config.TRAIN_SPLIT))
        neg_train = negative_patches[:neg_train_size]
        neg_val = negative_patches[neg_train_size:]
        
        # Combine positive and negative samples for final splits
        train_indices = pos_train + neg_train
        val_indices = pos_val + neg_val
        
        return train_indices, val_indices
    
    train_indices, val_indices = create_stratified_split()
    
    print(f"Train patches: {train_indices}")
    print(f"Val patches: {val_indices}")
    
    # Create data transforms with more aggressive augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])
    
    # Create datasets
    train_dataset = DeforestationDataset(train_indices, transform=train_transform, 
                                        use_sentinel1=use_sentinel1, use_sentinel2=use_sentinel2)
    val_dataset = DeforestationDataset(val_indices, transform=None,
                                     use_sentinel1=use_sentinel1, use_sentinel2=use_sentinel2)
    
    # Analyze class distribution
    train_pos_ratio = analyze_class_distribution(train_dataset, f"{model_name}_Train")
    val_pos_ratio = analyze_class_distribution(val_dataset, f"{model_name}_Val")
    
    # Create data loaders (disable pin_memory for CPU/MPS)
    use_pin_memory = config.DEVICE == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=use_pin_memory)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize model based on type
    if model_class == "LateFusion":
        model = LateFusionCNN().to(device)
    else:
        model = model_class(input_channels).to(device)
    
    # Use focal loss for better handling of class imbalance
    class FocalLoss(nn.Module):
        """
        Focal Loss implementation for handling class imbalance
        
        Focal Loss addresses the class imbalance problem by down-weighting
        easy examples and focusing on hard examples. This is particularly
        important for deforestation detection where deforested areas might
        be rare compared to forested areas.
        
        Formula: FL = -α(1-p_t)^γ * log(p_t)
        where p_t is the predicted probability for the true class
        """
        def __init__(self, alpha=1, gamma=2):
            super(FocalLoss, self).__init__()
            self.alpha = alpha  # Weighting factor for rare class
            self.gamma = gamma  # Focusing parameter (higher = more focus on hard examples)
            
        def forward(self, inputs, targets):
            # Calculate binary cross entropy loss
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss)  # Predicted probability for true class
            
            # Apply focal loss formula: down-weight easy examples
            focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
            return focal_loss.mean()
    
    # Adjust focal loss alpha based on class distribution
    # Higher alpha for more imbalanced datasets to give more weight to rare class
    alpha = 2.0 if train_pos_ratio < 0.3 else 1.0  # Higher alpha for imbalanced classes
    criterion = FocalLoss(alpha=alpha, gamma=2)
    print(f"Using Focal Loss with alpha={alpha} (class imbalance: {train_pos_ratio:.3f})")
    
    # Use AdamW optimizer with better weight decay for regularization
    # AdamW is Adam with decoupled weight decay, which often performs better than Adam
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    
    # Use cosine annealing with warm restarts for learning rate scheduling
    # This helps escape local minima and provides better convergence
    # T_0=5: restart every 5 epochs, T_mult=2: double restart period each time
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    # Training tracking variables
    train_losses = []      # Training loss history
    val_losses = []        # Validation loss history
    train_f1s = []        # Training F1 score history
    val_f1s = []          # Validation F1 score history
    learning_rates = []    # Learning rate history
    
    # Early stopping variables
    best_f1 = 0            # Best F1 score achieved so far
    best_epoch = 0        # Epoch when best F1 was achieved
    patience = 15         # Number of epochs to wait before early stopping
    patience_counter = 0  # Counter for patience
    
    print(f"\nStarting training for {model_name}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model type: {type(model).__name__}")
    
    # Main training loop
    for epoch in range(config.NUM_EPOCHS):
        # ========== TRAINING PHASE ==========
        model.train()  # Set model to training mode (enables dropout, batch norm updates, etc.)
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        # Iterate through training batches
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}"):
            # Move data to device (GPU/CPU) with non-blocking transfer for efficiency
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            
            # Forward pass: get model predictions
            outputs = model(images)
            
            # Calculate deforestation percentage for each patch
            # masks shape: (batch_size, 1, height, width)
            # mask_labels shape: (batch_size,) - deforestation rate per patch
            mask_labels = torch.mean(masks, dim=[2, 3])
            
            # Calculate loss using focal loss
            loss = criterion(outputs, mask_labels)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            # This is especially important for RNNs and deep networks
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update model parameters
            optimizer.step()
            
            # Store loss and predictions for metrics calculation
            total_loss += loss.item()
            all_predictions.append(outputs.detach())  # detach() prevents gradient computation
            all_targets.append(mask_labels.detach())
        
        # Calculate training metrics
        avg_loss = total_loss / len(train_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        # Use fixed threshold for training metrics (no data leakage)
        train_metrics = calculate_metrics(all_predictions, all_targets, threshold=0.5)
        train_losses.append(avg_loss)
        train_f1s.append(train_metrics[3])
        
        # ========== VALIDATION PHASE ==========
        model.eval()  # Set model to evaluation mode (disables dropout, freezes batch norm)
        val_total_loss = 0
        val_predictions = []
        val_targets = []
        
        # Validation loop with no gradient computation for efficiency
        with torch.no_grad():  # Disable gradient computation to save memory and speed
            for images, masks in val_loader:
                # Move data to device
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                
                # Forward pass only (no backward pass in validation)
                outputs = model(images)
                mask_labels = torch.mean(masks, dim=[2, 3])
                
                # Calculate loss for monitoring (but don't backpropagate)
                loss = criterion(outputs, mask_labels)
                val_total_loss += loss.item()
                
                # Store predictions and targets for metrics calculation
                val_predictions.append(outputs)
                val_targets.append(mask_labels)
        
        # Calculate validation metrics
        val_avg_loss = val_total_loss / len(val_loader)
        val_predictions = torch.cat(val_predictions)
        val_targets = torch.cat(val_targets)
        
        # Use debug mode for first 3 epochs to understand the issue
        debug_mode = epoch < 3
        
        # Use realistic fixed threshold (no data leakage)
        val_metrics = calculate_metrics(val_predictions, val_targets, threshold=0.5, debug=debug_mode)
        
        val_losses.append(val_avg_loss)
        val_f1s.append(val_metrics[3])
        
        # Debug: Print prediction and target ranges for validation
        val_pred_min = torch.min(val_predictions).item()
        val_pred_max = torch.max(val_predictions).item()
        val_pred_mean = torch.mean(val_predictions).item()
        val_target_min = torch.min(val_targets).item()
        val_target_max = torch.max(val_targets).item()
        val_target_mean = torch.mean(val_targets).item()
        
        print(f"         Val Pred range: [{val_pred_min:.4f}, {val_pred_max:.4f}], Mean: {val_pred_mean:.4f}")
        print(f"         Val Target range: [{val_target_min:.4f}, {val_target_max:.4f}], Mean: {val_target_mean:.4f}")
        print(f"         Val Metrics: Acc={val_metrics[0]:.4f}, Prec={val_metrics[1]:.4f}, Rec={val_metrics[2]:.4f}, F1={val_metrics[3]:.4f}")
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Print epoch results
        print(f"Epoch {epoch+1:3d}: Train Loss: {avg_loss:.4f}, Train F1: {train_metrics[3]:.4f}")
        print(f"         Val Loss: {val_avg_loss:.4f}, Val F1: {val_metrics[3]:.4f}")
        print(f"         LR: {current_lr:.6f}")
        
        # Save best model
        if val_metrics[3] > best_f1:
            best_f1 = val_metrics[3]
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), f"{model_name.lower()}_best.pth")
            print(f"         New best F1: {best_f1:.4f} (Epoch {best_epoch})")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience: {patience})")
            break
        
        # Check for overfitting and validation issues
        if epoch > 5 and len(train_f1s) > 3:
            recent_train_f1 = np.mean(train_f1s[-3:])
            recent_val_f1 = np.mean(val_f1s[-3:])
            if recent_train_f1 - recent_val_f1 > 0.4:  # Large gap indicates overfitting
                print(f"Warning: Potential overfitting detected (Train F1: {recent_train_f1:.3f}, Val F1: {recent_val_f1:.3f})")
                
        # Check for validation issues with small dataset
        if len(val_indices) < 3:
            print(f"Warning: Very small validation set ({len(val_indices)} samples). Results may not be reliable.")
            
        # Monitor learning rate
        if current_lr < 1e-6:
            print(f"Warning: Learning rate very low ({current_lr:.8f}). Training may be stuck.")
        
        print("-" * 60)
    
    print(f"\n{model_name} completed! Best F1: {best_f1:.4f} at epoch {best_epoch}")
    
    # Prepare results dictionary
    results = {
        'model_name': model_name,
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s,
        'learning_rates': learning_rates
    }
    
    # Save the trained model
    save_model_checkpoint(model, model_name, results, train_indices, val_indices, best_epoch)
    
    return results

def plot_training_results(results, save_path="training_results.png"):
    """Plot training and validation curves for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'red', 'green']
    
    for i, result in enumerate(results):
        epochs = range(1, len(result['train_losses']) + 1)
        color = colors[i % len(colors)]
        
        # Plot losses
        axes[0, 0].plot(epochs, result['train_losses'], label=f"{result['model_name']} Train", 
                       color=color, linestyle='-')
        axes[0, 0].plot(epochs, result['val_losses'], label=f"{result['model_name']} Val", 
                       color=color, linestyle='--')
        
        # Plot F1 scores
        axes[0, 1].plot(epochs, result['train_f1s'], label=f"{result['model_name']} Train", 
                       color=color, linestyle='-')
        axes[0, 1].plot(epochs, result['val_f1s'], label=f"{result['model_name']} Val", 
                       color=color, linestyle='--')
    
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('Training and Validation F1 Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot final results comparison
    model_names = [r['model_name'] for r in results]
    best_f1s = [r['best_f1'] for r in results]
    
    axes[1, 0].bar(model_names, best_f1s, color=colors[:len(model_names)])
    axes[1, 0].set_title('Best F1 Score Comparison')
    axes[1, 0].set_ylabel('Best F1 Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot training progress
    for i, result in enumerate(results):
        epochs = range(1, len(result['train_losses']) + 1)
        color = colors[i % len(colors)]
        axes[1, 1].plot(epochs, result['val_f1s'], label=result['model_name'], 
                       color=color, marker='o', markersize=3)
    
    axes[1, 1].set_title('Validation F1 Score Progress')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training results saved to {save_path}")

def main():
    """
    Main function that orchestrates the training of multiple deforestation detection models
    
    This function:
    1. Displays training configuration
    2. Checks for existing trained models
    3. Trains multiple model architectures with different data sources
    4. Visualizes results and provides final comparison
    
    Models trained:
    - SimpleCNN with Sentinel-2 (optical) data only
    - SimpleCNN with Sentinel-1 (radar) data only  
    - RobustBaselineCNN with combined data
    - LateFusionCNN with combined data
    """
    print("=" * 60)
    print("DEFORESTATION DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Max epochs: {config.NUM_EPOCHS}")
    print(f"Train/Val split: {config.TRAIN_SPLIT:.1f}/{1-config.TRAIN_SPLIT:.1f}")
    print("=" * 60)
    
    # Check if trained_models directory exists and show existing models
    if os.path.exists('trained_models'):
        existing_models = [f for f in os.listdir('trained_models') if f.endswith('_results.json')]
        if existing_models:
            print(f"\nFound {len(existing_models)} existing trained models:")
            for model_file in existing_models:
                model_name = model_file.replace('_results.json', '').replace('_', ' ').title()
                print(f"  - {model_name}")
            print("Models will be loaded if they exist, otherwise training will start.")
        else:
            print("\nNo existing models found. All models will be trained.")
    else:
        print("\nNo trained_models directory found. All models will be trained.")
    
    # Define models to train with their configurations
    # Format: (model_name, model_class, input_channels, use_sentinel1, use_sentinel2)
    models_to_train = [
        # Simple CNN with Sentinel-2 optical data only (4 bands: RGB + NIR)
        ("SimpleCNN_Sentinel2", SimpleCNN, config.SENTINEL2_BANDS, False, True),
        
        # Simple CNN with Sentinel-1 radar data only (2 bands: VV, VH)
        ("SimpleCNN_Sentinel1", SimpleCNN, config.SENTINEL1_BANDS, True, False),
        
        # Robust baseline CNN with combined data (6 bands total)
        ("RobustBaseline_Combined", RobustBaselineCNN, config.SENTINEL1_BANDS + config.SENTINEL2_BANDS, True, True),
        
        # Late fusion CNN with combined data (processes each source separately then fuses)
        ("LateFusion_Combined", "LateFusion", config.SENTINEL1_BANDS + config.SENTINEL2_BANDS, True, True)
    ]
    
    results = []
    
    for model_name, model_class, input_channels, use_s1, use_s2 in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"Model class: {model_class if isinstance(model_class, str) else model_class.__name__}")
        print(f"Input channels: {input_channels}")
        print(f"Using Sentinel-1: {use_s1}, Using Sentinel-2: {use_s2}")
        print(f"{'='*60}")
        
        # Check if model already exists
        if check_if_model_exists(model_name):
            print(f"✓ Model {model_name} already exists!")
            print("Loading existing model...")
            
            # Load existing results from JSON
            results_path = f'trained_models/{model_name.lower()}_results.json'
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    existing_results = json.load(f)
                print(f"Previous results - Best F1: {existing_results['best_f1']:.4f}")
                
                # Create results dict compatible with plotting
                result = {
                    'model_name': model_name,
                    'best_f1': existing_results['best_f1'],
                    'best_epoch': existing_results['best_epoch'],
                    'train_losses': [],  # We don't have this from JSON
                    'val_losses': [],
                    'train_f1s': [],
                    'val_f1s': [],
                    'learning_rates': []
                }
                results.append(result)
            else:
                print("Warning: Model exists but results file not found. Training anyway...")
                result = train_model(model_name, model_class, input_channels, use_s1, use_s2)
                results.append(result)
        else:
            print("Model doesn't exist. Starting training...")
            result = train_model(model_name, model_class, input_channels, use_s1, use_s2)
        results.append(result)
    
    # Plot results
    plot_training_results(results)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # Sort results by F1 score
    results_sorted = sorted(results, key=lambda x: x['best_f1'], reverse=True)
    
    for i, result in enumerate(results_sorted):
        rank = i + 1
        print(f"{rank}. {result['model_name']:15s}: F1 = {result['best_f1']:.4f}")
    
    print(f"\nBest performing model: {results_sorted[0]['model_name']}")
    print(f"Best F1 Score: {results_sorted[0]['best_f1']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
