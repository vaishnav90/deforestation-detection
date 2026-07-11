import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import os
import torchvision.transforms as transforms
import rasterio
import pickle
import json

import config
from data_loader import DeforestationDataset
from models import SimpleCNN, RobustBaselineCNN, LateFusionCNN

def calculate_metrics(predictions, targets, threshold=0.5, debug=False):
    pred_binary = (predictions > threshold).float()

    targets_binary = (targets > 0.1).float()

    pred_flat = pred_binary.cpu().numpy().flatten()
    target_flat = targets_binary.cpu().numpy().flatten()

    tp = np.sum((pred_flat == 1) & (target_flat == 1))
    fp = np.sum((pred_flat == 1) & (target_flat == 0))
    fn = np.sum((pred_flat == 0) & (target_flat == 1))
    tn = np.sum((pred_flat == 0) & (target_flat == 0))

    if debug:
        print(f"         Debug - Threshold: {threshold:.4f}")
        print(f"         Debug - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        print(f"         Debug - Pred binary: {pred_flat}")
        print(f"         Debug - Target binary: {target_flat}")
        print(f"         Debug - Raw predictions: {predictions.cpu().numpy().flatten()}")
        print(f"         Debug - Raw targets: {targets.cpu().numpy().flatten()}")

    total_samples = tp + fp + fn + tn
    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0

    accuracy = (tp + tn) / total_samples
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return accuracy, precision, recall, f1

def analyze_class_distribution(dataset, model_name):
    deforestation_rates = []
    for i in range(len(dataset)):
        _, mask = dataset[i]
        deforestation_rate = torch.mean(mask).item()
        deforestation_rates.append(deforestation_rate)

    deforestation_rates = np.array(deforestation_rates)

    positive_samples = np.sum(deforestation_rates > 0.1)
    total_samples = len(deforestation_rates)

    print(f"\n{model_name} Dataset Analysis:")
    print(f"Total samples: {total_samples}")
    print(f"Positive samples (deforestation > 10%): {positive_samples}")
    print(f"Negative samples: {total_samples - positive_samples}")
    print(f"Class ratio (pos:neg): {positive_samples}:{total_samples - positive_samples}")
    print(f"Mean deforestation rate: {np.mean(deforestation_rates):.4f}")
    print(f"Std deforestation rate: {np.std(deforestation_rates):.4f}")

    return positive_samples / total_samples if total_samples > 0 else 0

def save_model_checkpoint(model, model_name, results, train_indices, val_indices, epoch):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'results': results,
        'train_indices': train_indices,
        'val_indices': val_indices,
        'epoch': epoch,
        'config': {
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'num_epochs': config.NUM_EPOCHS,
            'device': config.DEVICE,
            'sentinel1_bands': config.SENTINEL1_BANDS,
            'sentinel2_bands': config.SENTINEL2_BANDS
        }
    }

    os.makedirs('trained_models', exist_ok=True)

    model_path = f'trained_models/{model_name.lower()}_best.pth'
    torch.save(model.state_dict(), model_path)

    checkpoint_path = f'trained_models/{model_name.lower()}_checkpoint.pkl'
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    results_path = f'trained_models/{model_name.lower()}_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'model_name': model_name,
            'best_f1': results['best_f1'],
            'best_epoch': results['best_epoch'],
            'final_train_loss': results['train_losses'][-1] if results['train_losses'] else 0,
            'final_val_loss': results['val_losses'][-1] if results['val_losses'] else 0,
            'final_train_f1': results['train_f1s'][-1] if results['train_f1s'] else 0,
            'final_val_f1': results['val_f1s'][-1] if results['val_f1s'] else 0,
            'total_epochs_trained': len(results['train_losses']),
            'train_indices': train_indices,
            'val_indices': val_indices
        }, f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Results saved to: {results_path}")

def load_model_checkpoint(model_name, model_class, input_channels):
    checkpoint_path = f'trained_models/{model_name.lower()}_checkpoint.pkl'

    if not os.path.exists(checkpoint_path):
        return None, None

    print(f"Loading model from: {checkpoint_path}")

    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    if model_class == "LateFusion":
        model = LateFusionCNN()
    else:
        model = model_class(input_channels)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint

def check_if_model_exists(model_name):
    checkpoint_path = f'trained_models/{model_name.lower()}_checkpoint.pkl'
    model_path = f'trained_models/{model_name.lower()}_best.pth'

    return os.path.exists(checkpoint_path) and os.path.exists(model_path)

def train_model(model_name, model_class, input_channels, use_sentinel1, use_sentinel2):
    device = torch.device(config.DEVICE)
    print(f"Training {model_name} on {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    def create_stratified_split():
        deforestation_rates = []
        for patch_idx in range(config.NUM_PATCHES):
            mask_path = f"{config.MASK_PATH}/RASTER_{patch_idx}.tif"
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                mask_binary = (mask == 1).astype(np.float32)
                deforestation_rate = np.mean(mask_binary)
                deforestation_rates.append(deforestation_rate)

        positive_patches = [i for i, rate in enumerate(deforestation_rates) if rate > 0.1]
        negative_patches = [i for i, rate in enumerate(deforestation_rates) if rate <= 0.1]

        np.random.shuffle(positive_patches)
        np.random.shuffle(negative_patches)

        pos_train_size = max(1, int(len(positive_patches) * config.TRAIN_SPLIT))
        pos_train = positive_patches[:pos_train_size]
        pos_val = positive_patches[pos_train_size:]

        neg_train_size = max(1, int(len(negative_patches) * config.TRAIN_SPLIT))
        neg_train = negative_patches[:neg_train_size]
        neg_val = negative_patches[neg_train_size:]

        train_indices = pos_train + neg_train
        val_indices = pos_val + neg_val

        return train_indices, val_indices

    train_indices, val_indices = create_stratified_split()

    print(f"Train patches: {train_indices}")
    print(f"Val patches: {val_indices}")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])

    train_dataset = DeforestationDataset(train_indices, transform=train_transform,
                                        use_sentinel1=use_sentinel1, use_sentinel2=use_sentinel2)
    val_dataset = DeforestationDataset(val_indices, transform=None,
                                     use_sentinel1=use_sentinel1, use_sentinel2=use_sentinel2)

    train_pos_ratio = analyze_class_distribution(train_dataset, f"{model_name}_Train")
    val_pos_ratio = analyze_class_distribution(val_dataset, f"{model_name}_Val")

    use_pin_memory = config.DEVICE == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=use_pin_memory)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    if model_class == "LateFusion":
        model = LateFusionCNN().to(device)
    else:
        model = model_class(input_channels).to(device)

    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss)

            focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
            return focal_loss.mean()

    alpha = 2.0 if train_pos_ratio < 0.3 else 1.0
    criterion = FocalLoss(alpha=alpha, gamma=2)
    print(f"Using Focal Loss with alpha={alpha} (class imbalance: {train_pos_ratio:.3f})")

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    train_losses = []
    val_losses = []
    train_f1s = []
    val_f1s = []
    learning_rates = []

    best_f1 = 0
    best_epoch = 0
    patience = 15
    patience_counter = 0

    print(f"\nStarting training for {model_name}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model type: {type(model).__name__}")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}"):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)

            mask_labels = torch.mean(masks, dim=[2, 3])

            loss = criterion(outputs, mask_labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            all_predictions.append(outputs.detach())
            all_targets.append(mask_labels.detach())

        avg_loss = total_loss / len(train_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        train_metrics = calculate_metrics(all_predictions, all_targets, threshold=0.5)
        train_losses.append(avg_loss)
        train_f1s.append(train_metrics[3])

        model.eval()
        val_total_loss = 0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

                outputs = model(images)
                mask_labels = torch.mean(masks, dim=[2, 3])

                loss = criterion(outputs, mask_labels)
                val_total_loss += loss.item()

                val_predictions.append(outputs)
                val_targets.append(mask_labels)

        val_avg_loss = val_total_loss / len(val_loader)
        val_predictions = torch.cat(val_predictions)
        val_targets = torch.cat(val_targets)

        debug_mode = epoch < 3

        val_metrics = calculate_metrics(val_predictions, val_targets, threshold=0.5, debug=debug_mode)

        val_losses.append(val_avg_loss)
        val_f1s.append(val_metrics[3])

        val_pred_min = torch.min(val_predictions).item()
        val_pred_max = torch.max(val_predictions).item()
        val_pred_mean = torch.mean(val_predictions).item()
        val_target_min = torch.min(val_targets).item()
        val_target_max = torch.max(val_targets).item()
        val_target_mean = torch.mean(val_targets).item()

        print(f"         Val Pred range: [{val_pred_min:.4f}, {val_pred_max:.4f}], Mean: {val_pred_mean:.4f}")
        print(f"         Val Target range: [{val_target_min:.4f}, {val_target_max:.4f}], Mean: {val_target_mean:.4f}")
        print(f"         Val Metrics: Acc={val_metrics[0]:.4f}, Prec={val_metrics[1]:.4f}, Rec={val_metrics[2]:.4f}, F1={val_metrics[3]:.4f}")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        print(f"Epoch {epoch+1:3d}: Train Loss: {avg_loss:.4f}, Train F1: {train_metrics[3]:.4f}")
        print(f"         Val Loss: {val_avg_loss:.4f}, Val F1: {val_metrics[3]:.4f}")
        print(f"         LR: {current_lr:.6f}")

        if val_metrics[3] > best_f1:
            best_f1 = val_metrics[3]
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), f"{model_name.lower()}_best.pth")
            print(f"         New best F1: {best_f1:.4f} (Epoch {best_epoch})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience: {patience})")
            break

        if epoch > 5 and len(train_f1s) > 3:
            recent_train_f1 = np.mean(train_f1s[-3:])
            recent_val_f1 = np.mean(val_f1s[-3:])
            if recent_train_f1 - recent_val_f1 > 0.4:
                print(f"Warning: Potential overfitting detected (Train F1: {recent_train_f1:.3f}, Val F1: {recent_val_f1:.3f})")

        if len(val_indices) < 3:
            print(f"Warning: Very small validation set ({len(val_indices)} samples). Results may not be reliable.")

        if current_lr < 1e-6:
            print(f"Warning: Learning rate very low ({current_lr:.8f}). Training may be stuck.")

        print("-" * 60)

    print(f"\n{model_name} completed! Best F1: {best_f1:.4f} at epoch {best_epoch}")

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

    save_model_checkpoint(model, model_name, results, train_indices, val_indices, best_epoch)

    return results

def plot_training_results(results, save_path="training_results.png"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    colors = ['blue', 'red', 'green']

    for i, result in enumerate(results):
        epochs = range(1, len(result['train_losses']) + 1)
        color = colors[i % len(colors)]

        axes[0, 0].plot(epochs, result['train_losses'], label=f"{result['model_name']} Train",
                       color=color, linestyle='-')
        axes[0, 0].plot(epochs, result['val_losses'], label=f"{result['model_name']} Val",
                       color=color, linestyle='--')

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

    model_names = [r['model_name'] for r in results]
    best_f1s = [r['best_f1'] for r in results]

    axes[1, 0].bar(model_names, best_f1s, color=colors[:len(model_names)])
    axes[1, 0].set_title('Best F1 Score Comparison')
    axes[1, 0].set_ylabel('Best F1 Score')
    axes[1, 0].tick_params(axis='x', rotation=45)

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
    print("=" * 60)
    print("DEFORESTATION DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Max epochs: {config.NUM_EPOCHS}")
    print(f"Train/Val split: {config.TRAIN_SPLIT:.1f}/{1-config.TRAIN_SPLIT:.1f}")
    print("=" * 60)

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

    models_to_train = [

        ("SimpleCNN_Sentinel2", SimpleCNN, config.SENTINEL2_BANDS, False, True),

        ("SimpleCNN_Sentinel1", SimpleCNN, config.SENTINEL1_BANDS, True, False),

        ("RobustBaseline_Combined", RobustBaselineCNN, config.SENTINEL1_BANDS + config.SENTINEL2_BANDS, True, True),

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

        if check_if_model_exists(model_name):
            print(f"✓ Model {model_name} already exists!")
            print("Loading existing model...")

            results_path = f'trained_models/{model_name.lower()}_results.json'
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    existing_results = json.load(f)
                print(f"Previous results - Best F1: {existing_results['best_f1']:.4f}")

                result = {
                    'model_name': model_name,
                    'best_f1': existing_results['best_f1'],
                    'best_epoch': existing_results['best_epoch'],
                    'train_losses': [],
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

    plot_training_results(results)

    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")

    results_sorted = sorted(results, key=lambda x: x['best_f1'], reverse=True)

    for i, result in enumerate(results_sorted):
        rank = i + 1
        print(f"{rank}. {result['model_name']:15s}: F1 = {result['best_f1']:.4f}")

    print(f"\nBest performing model: {results_sorted[0]['model_name']}")
    print(f"Best F1 Score: {results_sorted[0]['best_f1']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
