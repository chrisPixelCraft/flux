"""
BezierFontDataset for loading preprocessed font images and Bézier curves.

Implements efficient data loading with caching, augmentation, and proper batching
for the BezierAdapter training pipeline.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as transforms

from ..modules.models import BezierControlPoints, MultiModalCondition


@dataclass
class BezierFontSample:
    """Single training sample with all modalities."""
    image_path: str
    style_image_path: str
    mask_path: str
    bezier_points: List[Tuple[float, float]]
    character: str
    font_name: str
    style_prompt: str


class BezierFontDataset(Dataset):
    """
    Dataset for BezierAdapter training with font images and Bézier curve annotations.
    
    Loads preprocessed data from bezier_extraction.py output and provides
    multi-modal samples for training the BezierAdapter framework.
    """
    
    def __init__(self,
                 data_root: str,
                 split: str = "train",
                 image_size: int = 512,
                 latent_size: int = 64,
                 augment: bool = True,
                 cache_size: int = 1000,
                 min_bezier_points: int = 4,
                 is_fill_model: bool = True,
                 mask_channels: int = 320):
        """
        Initialize BezierFontDataset.
        
        Args:
            data_root: Path to directory containing preprocessed font data
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image resolution for style images
            latent_size: Target latent resolution (FLUX VAE latent space)
            augment: Whether to apply data augmentation
            cache_size: Number of samples to keep in memory cache
            min_bezier_points: Minimum number of Bézier control points required
            is_fill_model: Whether using FLUX.1-Fill-dev (384 channels)
            mask_channels: Number of mask conditioning channels (4 for standard, 320 for Fill)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.latent_size = latent_size
        self.augment = augment
        self.cache_size = cache_size
        self.min_bezier_points = min_bezier_points
        self.is_fill_model = is_fill_model
        self.mask_channels = mask_channels
        
        # Load dataset metadata
        self.samples = self._load_samples()
        
        # Initialize transforms
        self._setup_transforms()
        
        # Memory cache for frequently accessed samples
        self.cache = {}
        self.cache_order = []
        
        # Style prompts for text conditioning
        self.style_prompts = [
            "elegant serif typography",
            "modern sans-serif font",
            "handwritten calligraphy style",
            "bold display lettering",
            "vintage typeface design",
            "minimalist clean typography",
            "decorative ornamental font",
            "technical monospace letters"
        ]
    
    def _load_samples(self) -> List[BezierFontSample]:
        """Load sample metadata from preprocessed data directory."""
        samples = []
        
        # Look for metadata file from bezier_extraction.py
        metadata_file = self.data_root / f"{self.split}_metadata.json"
        
        if metadata_file.exists():
            # Load from existing metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            for item in metadata:
                if len(item['bezier_points']) >= self.min_bezier_points:
                    samples.append(BezierFontSample(**item))
        else:
            # Scan directory structure (fallback)
            samples = self._scan_directory_structure()
        
        print(f"Loaded {len(samples)} samples for {self.split} split")
        return samples
    
    def _scan_directory_structure(self) -> List[BezierFontSample]:
        """Scan directory structure to find font samples (fallback method)."""
        samples = []
        
        # Expected structure: data_root/font_name/character/
        for font_dir in self.data_root.iterdir():
            if not font_dir.is_dir():
                continue
                
            for char_dir in font_dir.iterdir():
                if not char_dir.is_dir():
                    continue
                    
                # Look for required files
                image_path = char_dir / "rendered.png"
                style_path = char_dir / "style.png"
                mask_path = char_dir / "mask.png"
                bezier_path = char_dir / "bezier.json"
                
                if all(p.exists() for p in [image_path, style_path, mask_path, bezier_path]):
                    # Load Bézier points
                    with open(bezier_path, 'r') as f:
                        bezier_data = json.load(f)
                    
                    if len(bezier_data['control_points']) >= self.min_bezier_points:
                        samples.append(BezierFontSample(
                            image_path=str(image_path),
                            style_image_path=str(style_path),
                            mask_path=str(mask_path),
                            bezier_points=bezier_data['control_points'],
                            character=char_dir.name,
                            font_name=font_dir.name,
                            style_prompt=random.choice(self.style_prompts)
                        ))
        
        return samples
    
    def _setup_transforms(self):
        """Setup image transformations for training."""
        if self.augment:
            # Augmentation transforms for style images
            self.style_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.RandomRotation(degrees=2),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] range
            ])
            
            # Augmentation for mask images
            # Note: For Fill model, masks will be processed differently to handle 320 channels
            if self.is_fill_model:
                # Fill model uses larger mask conditioning - resize to intermediate size first
                self.mask_transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),  # Full resolution first
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ToTensor()
                ])
            else:
                # Standard FLUX uses VAE latent space size
                self.mask_transform = transforms.Compose([
                    transforms.Resize((self.latent_size, self.latent_size)),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ToTensor()
                ])
        else:
            # No augmentation transforms
            self.style_transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
            # No augmentation for mask images
            if self.is_fill_model:
                # Fill model uses full resolution masks
                self.mask_transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor()
                ])
            else:
                # Standard FLUX uses VAE latent space size
                self.mask_transform = transforms.Compose([
                    transforms.Resize((self.latent_size, self.latent_size)),
                    transforms.ToTensor()
                ])
    
    def _augment_bezier_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Apply random augmentation to Bézier control points."""
        if not self.augment:
            return points
            
        augmented = []
        
        # Random perturbation parameters
        translation_range = 0.05  # 5% of coordinate space
        scale_range = 0.1        # ±10% scaling
        
        # Random global transformations
        tx = random.uniform(-translation_range, translation_range)
        ty = random.uniform(-translation_range, translation_range)
        scale = random.uniform(1.0 - scale_range, 1.0 + scale_range)
        
        for x, y in points:
            # Apply transformations
            new_x = (x * scale + tx)
            new_y = (y * scale + ty)
            
            # Clamp to valid range [0, 1]
            new_x = max(0.0, min(1.0, new_x))
            new_y = max(0.0, min(1.0, new_y))
            
            augmented.append((new_x, new_y))
        
        return augmented
    
    def _load_and_process_image(self, image_path: str, transform: transforms.Compose) -> Tensor:
        """Load and process image with caching."""
        cache_key = (image_path, id(transform))
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        processed = transform(image)
        
        # Add to cache if space available
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = processed
            self.cache_order.append(cache_key)
        elif self.cache_order:  # FIFO cache eviction
            old_key = self.cache_order.pop(0)
            if old_key in self.cache:
                del self.cache[old_key]
            self.cache[cache_key] = processed
            self.cache_order.append(cache_key)
        
        return processed
    
    def _prepare_fill_model_mask(self, mask_tensor: Tensor, style_image: Tensor) -> Tensor:
        """
        Prepare extended mask conditioning for FLUX.1-Fill-dev model.
        
        Fill models expect 320 channels: VAE-encoded image (316) + mask (4)
        This method creates a simulated extended conditioning similar to Fill model's format.
        
        Args:
            mask_tensor: Original mask tensor (1, H, W)
            style_image: Style reference image (3, H, W)
            
        Returns:
            Extended mask conditioning tensor (320, H_latent, W_latent)
        """
        # Simulate VAE encoding of the style image (316 channels)
        # In practice, this would use the actual VAE encoder
        H_latent, W_latent = self.latent_size, self.latent_size
        
        # Create simulated VAE latent features (316 channels)
        vae_features = torch.randn(316, H_latent, W_latent)
        
        # Resize and expand mask to latent space (4 channels for consistency)
        mask_resized = F.interpolate(
            mask_tensor.unsqueeze(0), 
            size=(H_latent, W_latent), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        # Expand mask to 4 channels (standard VAE latent space)
        mask_4ch = mask_resized.repeat(4, 1, 1)
        
        # Combine VAE features + mask = 320 channels total
        extended_conditioning = torch.cat([vae_features, mask_4ch], dim=0)
        
        return extended_conditioning
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single training sample.
        
        Returns:
            Dictionary containing all modalities for training:
            - style_image: Style reference image (3, H, W)
            - mask_image: Character mask (1, H_latent, W_latent)  
            - bezier_points: Normalized Bézier control points (N, 2)
            - character: Target character string
            - style_prompt: Text description for style
        """
        sample = self.samples[idx]
        
        # Load and process style image
        style_image = self._load_and_process_image(sample.style_image_path, self.style_transform)
        
        # Load and process mask image
        mask_image = Image.open(sample.mask_path).convert("L")
        mask_tensor = self.mask_transform(mask_image)
        
        # For Fill model, expand mask to match conditioning channels (320)
        if self.is_fill_model:
            # Create extended mask conditioning for Fill model
            # This simulates the VAE+mask encoding that Fill model expects
            mask_tensor = self._prepare_fill_model_mask(mask_tensor, style_image)
        
        # Process Bézier points with augmentation
        bezier_points = self._augment_bezier_points(sample.bezier_points)
        bezier_tensor = torch.tensor(bezier_points, dtype=torch.float32)
        
        # Add density values to Bézier points (computed from KDE)
        # For now, use uniform density - will be computed by BezierParameterProcessor
        num_points = bezier_tensor.shape[0]
        densities = torch.ones(num_points, 1) / num_points  # Uniform density
        bezier_features = torch.cat([bezier_tensor, densities], dim=1)  # (N, 3)
        
        return {
            'style_image': style_image,
            'mask_image': mask_tensor,
            'bezier_points': bezier_tensor,
            'bezier_features': bezier_features,
            'character': sample.character,
            'font_name': sample.font_name,
            'style_prompt': sample.style_prompt,
            'sample_id': f"{sample.font_name}_{sample.character}_{idx}"
        }


class DataCollator:
    """
    Collates batches of BezierFontDataset samples with proper padding and batching.
    
    Handles variable-length Bézier point sequences and ensures consistent
    batch dimensions across all modalities.
    """
    
    def __init__(self, 
                 max_bezier_points: int = 32,
                 pad_token: float = -1.0):
        """
        Initialize DataCollator.
        
        Args:
            max_bezier_points: Maximum number of Bézier points per sample
            pad_token: Value used for padding Bézier point sequences
        """
        self.max_bezier_points = max_bezier_points
        self.pad_token = pad_token
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """
        Collate batch of samples into tensors.
        
        Args:
            batch: List of sample dictionaries from BezierFontDataset
            
        Returns:
            Dictionary of batched tensors ready for training
        """
        batch_size = len(batch)
        
        # Stack regular tensors
        style_images = torch.stack([sample['style_image'] for sample in batch])
        mask_images = torch.stack([sample['mask_image'] for sample in batch])
        
        # Handle variable-length Bézier sequences with padding
        bezier_points_list = [sample['bezier_points'] for sample in batch]
        bezier_features_list = [sample['bezier_features'] for sample in batch]
        
        # Pad Bézier sequences to max length
        padded_points = []
        padded_features = []
        bezier_masks = []  # Mask for valid (non-padded) points
        
        for points, features in zip(bezier_points_list, bezier_features_list):
            num_points = points.shape[0]
            
            if num_points > self.max_bezier_points:
                # Truncate if too long
                points = points[:self.max_bezier_points]
                features = features[:self.max_bezier_points]
                mask = torch.ones(self.max_bezier_points, dtype=torch.bool)
            else:
                # Pad if too short
                pad_length = self.max_bezier_points - num_points
                
                # Pad points and features
                point_padding = torch.full((pad_length, 2), self.pad_token)
                feature_padding = torch.full((pad_length, 3), self.pad_token)
                
                points = torch.cat([points, point_padding], dim=0)
                features = torch.cat([features, feature_padding], dim=0)
                
                # Create mask (True for valid points, False for padding)
                mask = torch.cat([
                    torch.ones(num_points, dtype=torch.bool),
                    torch.zeros(pad_length, dtype=torch.bool)
                ], dim=0)
            
            padded_points.append(points)
            padded_features.append(features)
            bezier_masks.append(mask)
        
        # Stack padded sequences
        bezier_points_batch = torch.stack(padded_points)
        bezier_features_batch = torch.stack(padded_features)
        bezier_masks_batch = torch.stack(bezier_masks)
        
        # Collect text data
        characters = [sample['character'] for sample in batch]
        font_names = [sample['font_name'] for sample in batch]
        style_prompts = [sample['style_prompt'] for sample in batch]
        sample_ids = [sample['sample_id'] for sample in batch]
        
        return {
            'style_images': style_images,           # (B, 3, H, W)
            'mask_images': mask_images,             # (B, 1, H_latent, W_latent)
            'bezier_points': bezier_points_batch,   # (B, max_points, 2)
            'bezier_features': bezier_features_batch, # (B, max_points, 3)
            'bezier_masks': bezier_masks_batch,     # (B, max_points)
            'characters': characters,               # List[str]
            'font_names': font_names,               # List[str]
            'style_prompts': style_prompts,         # List[str]
            'sample_ids': sample_ids                # List[str]
        }


def create_dataloader(dataset: BezierFontDataset,
                     batch_size: int = 4,
                     shuffle: bool = True,
                     num_workers: int = 4,
                     pin_memory: bool = True,
                     max_bezier_points: int = 32) -> DataLoader:
    """
    Create DataLoader for BezierFontDataset with proper collation.
    
    Args:
        dataset: BezierFontDataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle samples
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        max_bezier_points: Maximum Bézier points per sample
        
    Returns:
        Configured DataLoader
    """
    collator = DataCollator(max_bezier_points=max_bezier_points)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=True  # Ensure consistent batch sizes
    )


def split_dataset(data_root: str, 
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 seed: int = 42,
                 is_fill_model: bool = True,
                 mask_channels: int = 320) -> Tuple[BezierFontDataset, BezierFontDataset, BezierFontDataset]:
    """
    Create train/val/test splits from data directory.
    
    Args:
        data_root: Path to preprocessed font data
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        seed: Random seed for reproducible splits
        is_fill_model: Whether using FLUX.1-Fill-dev model
        mask_channels: Number of mask conditioning channels
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Create full dataset to get sample list
    full_dataset = BezierFontDataset(
        data_root, 
        split="full", 
        augment=False, 
        is_fill_model=is_fill_model, 
        mask_channels=mask_channels
    )
    
    # Set random seed for reproducible splits
    random.seed(seed)
    samples = full_dataset.samples.copy()
    random.shuffle(samples)
    
    # Calculate split indices
    total_samples = len(samples)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    # Split samples
    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]
    
    # Create datasets with appropriate splits
    train_dataset = BezierFontDataset(
        data_root, 
        split="train", 
        augment=True, 
        is_fill_model=is_fill_model, 
        mask_channels=mask_channels
    )
    train_dataset.samples = train_samples
    
    val_dataset = BezierFontDataset(
        data_root, 
        split="val", 
        augment=False, 
        is_fill_model=is_fill_model, 
        mask_channels=mask_channels
    )
    val_dataset.samples = val_samples
    
    test_dataset = BezierFontDataset(
        data_root, 
        split="test", 
        augment=False, 
        is_fill_model=is_fill_model, 
        mask_channels=mask_channels
    )
    test_dataset.samples = test_samples
    
    print(f"Dataset split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    
    return train_dataset, val_dataset, test_dataset