"""
NEURAL NETWORK MODELS FOR DEFORESTATION DETECTION
=================================================

This module contains various CNN architectures for deforestation detection
using satellite imagery. Each model is designed for different use cases:

1. SimpleCNN: Basic CNN with regularization - good baseline
2. RobustBaselineCNN: Advanced CNN with residual blocks - robust performance
3. LateFusionCNN: Multi-modal fusion - combines different data sources

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for deforestation detection
    
    This is a basic CNN architecture with:
    - Multiple convolutional layers with increasing filters
    - Batch normalization for stable training
    - Dropout for regularization
    - Global average pooling to reduce overfitting
    - Fully connected layers for classification
    
    Architecture:
    Input -> Conv7x7 -> Conv5x5 -> Conv3x3 -> Conv3x3 -> GlobalPool -> FC -> FC -> Output
    """
    
    def __init__(self, input_channels, num_classes=1):
        """
        Initialize SimpleCNN
        
        Args:
            input_channels: Number of input channels (depends on data sources)
                           - Sentinel-1 only: 2 channels
                           - Sentinel-2 only: 4 channels  
                           - Combined: 6 channels
            num_classes: Number of output classes (1 for binary classification)
        """
        super(SimpleCNN, self).__init__()
        
        # Feature extraction layers with progressive downsampling
        # Layer 1: Large receptive field, aggressive downsampling
        self.conv1 = nn.Conv2d(input_channels, 32, 7, stride=4, padding=3)
        self.bn1 = nn.BatchNorm2d(32)  # Normalize activations for stable training
        self.dropout1 = nn.Dropout2d(0.1)  # Prevent overfitting
        
        # Layer 2: Medium receptive field
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.1)
        
        # Layer 3: Small receptive field
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(0.1)
        
        # Layer 4: Additional feature extraction without downsampling
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout2d(0.1)
        
        # Global average pooling: reduces spatial dimensions to 1x1
        # This helps prevent overfitting and makes the model more robust
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier: fully connected layers
        self.fc1 = nn.Linear(256, 128)  # First FC layer
        self.fc2 = nn.Linear(128, 64)   # Second FC layer
        self.fc3 = nn.Linear(64, num_classes)  # Output layer
        
        # Dropout for fully connected layers
        self.dropout_fc = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) - deforestation probability
        """
        # Feature extraction with progressive downsampling
        # Each layer reduces spatial dimensions while increasing feature channels
        
        # Layer 1: Large receptive field (7x7), stride 4 for aggressive downsampling
        x = F.relu(self.bn1(self.conv1(x)))  # Apply conv, batch norm, then ReLU
        x = self.dropout1(x)  # Apply spatial dropout
        
        # Layer 2: Medium receptive field (5x5), stride 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        
        # Layer 3: Small receptive field (3x3), stride 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        
        # Layer 4: Additional feature extraction without downsampling
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout4(x)
        
        # Global average pooling: convert spatial features to single values
        # This makes the model invariant to spatial location and reduces overfitting
        x = self.global_pool(x)  # Shape: (batch_size, 256, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 256)
        
        # Classification layers
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.dropout_fc(x)   # Dropout for regularization
        
        x = F.relu(self.fc2(x))  # Second fully connected layer
        x = self.dropout_fc(x)   # Dropout for regularization
        
        # Output layer with sigmoid activation for binary classification
        # Sigmoid outputs values between 0 and 1 (deforestation probability)
        x = torch.sigmoid(self.fc3(x))
        
        return x

class UNetLike(nn.Module):
    def __init__(self, input_channels, num_classes=1):
        super(UNetLike, self).__init__()
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        
        d3 = self.decoder3(e3)
        d2 = self.decoder2(d3)
        
        out = torch.sigmoid(self.final(d2))
        return out

class RobustBaselineCNN(nn.Module):
    """
    Robust Baseline CNN with residual connections and strong regularization
    
    This model is designed for better performance and stability:
    - Residual blocks for easier training of deep networks
    - Max pooling for spatial downsampling
    - Strong regularization with batch norm and dropout
    - Adaptive average pooling for robustness
    
    Architecture:
    Input -> Conv7x7 -> MaxPool -> ResidualBlock -> Downsample -> ResidualBlock -> GlobalPool -> Classifier
    """
    
    def __init__(self, input_channels, num_classes=1):
        """
        Initialize RobustBaselineCNN
        
        Args:
            input_channels: Number of input channels (2, 4, or 6)
            num_classes: Number of output classes (1 for binary classification)
        """
        super(RobustBaselineCNN, self).__init__()
        
        # Initial feature extraction with aggressive downsampling
        self.conv1 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)  # Additional downsampling
        
        # Residual block 1: learn residual features
        # Residual connections help with gradient flow in deep networks
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Downsample to increase feature channels
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Residual block 2
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier with strong regularization
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Initial feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)
        
        # Residual block 1
        residual = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += residual
        x = F.relu(x)
        
        # Downsample
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Residual block 2
        residual = x
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.bn6(self.conv6(x))
        x += residual
        x = F.relu(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

class LateFusionCNN(nn.Module):
    """
    Late Fusion CNN for multi-modal deforestation detection
    
    This model processes Sentinel-1 (radar) and Sentinel-2 (optical) data separately
    in dedicated branches, then fuses the features at the end. This approach:
    - Allows each branch to specialize in its data type
    - Captures modality-specific features
    - Fuses information at a high level
    
    Architecture:
    Input -> Split -> Sentinel1_Branch -> Features1 -> Concatenate -> Fusion -> Output
                  -> Sentinel2_Branch -> Features2 ->
    """
    
    def __init__(self, sentinel1_channels=2, sentinel2_channels=4, num_classes=1):
        """
        Initialize LateFusionCNN
        
        Args:
            sentinel1_channels: Number of Sentinel-1 channels (2: VV, VH)
            sentinel2_channels: Number of Sentinel-2 channels (4: RGB + NIR)
            num_classes: Number of output classes (1 for binary classification)
        """
        super(LateFusionCNN, self).__init__()
        
        # Sentinel-1 branch: specialized for radar data
        # Radar data has different characteristics than optical data
        self.sentinel1_branch = nn.Sequential(
            nn.Conv2d(sentinel1_channels, 32, 3, padding=1),  # Extract radar features
            nn.BatchNorm2d(32),  # Normalize activations
            nn.ReLU(inplace=True),  # Non-linear activation
            nn.Dropout2d(0.1),  # Spatial dropout for regularization
            
            nn.Conv2d(32, 64, 3, padding=1),  # Deeper feature extraction
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.AdaptiveAvgPool2d((1, 1))  # Global pooling: (batch, 64, 1, 1)
        )
        
        # Sentinel-2 branch: specialized for optical data
        # Optical data provides color and spectral information
        self.sentinel2_branch = nn.Sequential(
            nn.Conv2d(sentinel2_channels, 32, 3, padding=1),  # Extract optical features
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, 3, padding=1),  # Deeper feature extraction
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.AdaptiveAvgPool2d((1, 1))  # Global pooling: (batch, 64, 1, 1)
        )
        
        # Fusion network: combines features from both modalities
        # Input: 128 features (64 from each branch)
        self.fusion = nn.Sequential(
            nn.Linear(128, 64),  # First fusion layer
            nn.BatchNorm1d(64),  # Batch normalization
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # High dropout for regularization
            
            nn.Linear(64, 32),   # Second fusion layer
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Moderate dropout
            
            nn.Linear(32, num_classes),  # Output layer
            nn.Sigmoid()  # Sigmoid for binary classification
        )
        
    def forward(self, x):
        """
        Forward pass through the late fusion network
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               Expected to contain both Sentinel-1 and Sentinel-2 data concatenated
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) - deforestation probability
        """
        # Split input into Sentinel-1 and Sentinel-2 components
        # Sentinel-1: first 2 channels (VV, VH polarizations)
        sentinel1 = x[:, :config.SENTINEL1_BANDS, :, :]
        
        # Sentinel-2: remaining channels (RGB + NIR)
        sentinel2 = x[:, config.SENTINEL1_BANDS:, :, :]
        
        # Process each modality through its specialized branch
        # Each branch extracts modality-specific features
        s1_features = self.sentinel1_branch(sentinel1).view(x.size(0), -1)  # Shape: (batch, 64)
        s2_features = self.sentinel2_branch(sentinel2).view(x.size(0), -1)  # Shape: (batch, 64)
        
        # Concatenate features from both modalities
        # This creates a combined representation of both radar and optical information
        combined = torch.cat([s1_features, s2_features], dim=1)  # Shape: (batch, 128)
        
        # Fuse the combined features through the fusion network
        out = self.fusion(combined)  # Shape: (batch, 1)
        
        return out
