#!/usr/bin/env python3
"""
Comprehensive EDA and Data Cleaning Script
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Image processing
import cv2
from PIL import Image
from collections import Counter, defaultdict
from tqdm import tqdm
import json

# Machine Learning
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

def analyze_dataset_structure():
    """Analyze the structure of the processed dataset"""
    
    print("📊 DATASET STRUCTURE ANALYSIS")
    print("=" * 60)
    
    base_path = Path("/Users/debabratapattnayak/web-dev/greencast")
    processed_data_path = base_path / "processed_data"
    
    # Choose dataset
    symlink_path = processed_data_path / "plantvillage_color_symlinks"
    copy_path = processed_data_path / "plantvillage_color"
    
    if symlink_path.exists():
        dataset_path = symlink_path
        print(f"📁 Using symlinked dataset")
    elif copy_path.exists():
        dataset_path = copy_path
        print(f"📁 Using copied dataset")
    else:
        print("❌ No processed dataset found!")
        return None
    
    # Analyze splits
    splits = ['train', 'validation', 'test']
    dataset_info = {}
    
    for split in splits:
        split_path = dataset_path / split
        if split_path.exists():
            classes = [d.name for d in split_path.iterdir() if d.is_dir()]
            
            split_info = {}
            total_images = 0
            
            for class_name in classes:
                class_path = split_path / class_name
                # Count images (handle different extensions)
                image_count = 0
                for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
                    image_count += len(list(class_path.glob(ext)))
                
                split_info[class_name] = image_count
                total_images += image_count
            
            dataset_info[split] = {
                'classes': split_info,
                'total_images': total_images,
                'num_classes': len(classes)
            }
            
            print(f"\n📈 {split.upper()} SET:")
            print(f"  Classes: {len(classes)}")
            print(f"  Total Images: {total_images:,}")
    
    return dataset_info, dataset_path

def create_class_distribution_analysis(dataset_info):
    """Create comprehensive class distribution analysis"""
    
    print("\n📊 CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Combine all splits for overall analysis
    all_classes = {}
    
    for split, info in dataset_info.items():
        for class_name, count in info['classes'].items():
            if class_name not in all_classes:
                all_classes[class_name] = {'train': 0, 'validation': 0, 'test': 0, 'total': 0}
            all_classes[class_name][split] = count
            all_classes[class_name]['total'] += count
    
    # Create DataFrame for analysis
    df_classes = pd.DataFrame(all_classes).T
    df_classes = df_classes.fillna(0).astype(int)
    df_classes = df_classes.sort_values('total', ascending=False)
    
    print(f"📋 Dataset Summary:")
    print(f"  Total Classes: {len(df_classes)}")
    print(f"  Total Images: {df_classes['total'].sum():,}")
    print(f"  Average per class: {df_classes['total'].mean():.0f}")
    print(f"  Min per class: {df_classes['total'].min()}")
    print(f"  Max per class: {df_classes['total'].max()}")
    
    # Class imbalance analysis
    print(f"\n⚖️ Class Balance Analysis:")
    max_count = df_classes['total'].max()
    min_count = df_classes['total'].min()
    imbalance_ratio = max_count / min_count
    print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 2:
        print(f"  ⚠️  Dataset is imbalanced (ratio > 2:1)")
    else:
        print(f"  ✅ Dataset is relatively balanced")
    
    # Top and bottom classes
    print(f"\n🔝 Top 5 Classes by Count:")
    for i, (class_name, row) in enumerate(df_classes.head().iterrows(), 1):
        print(f"  {i}. {class_name}: {row['total']:,} images")
    
    print(f"\n🔻 Bottom 5 Classes by Count:")
    for i, (class_name, row) in enumerate(df_classes.tail().iterrows(), 1):
        print(f"  {i}. {class_name}: {row['total']:,} images")
    
    return df_classes

def analyze_image_properties(dataset_path, sample_size=100):
    """Analyze image properties (size, format, quality)"""
    
    print(f"\n🖼️ IMAGE PROPERTIES ANALYSIS")
    print("=" * 60)
    
    # Sample images from train set
    train_path = dataset_path / 'train'
    classes = [d.name for d in train_path.iterdir() if d.is_dir()]
    
    image_info = []
    sample_per_class = max(1, sample_size // len(classes))
    
    print(f"📸 Analyzing {sample_per_class} images per class...")
    
    for class_name in tqdm(classes[:10], desc="Analyzing classes"):  # Limit to first 10 classes
        class_path = train_path / class_name
        
        # Get sample images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
            image_files.extend(list(class_path.glob(ext)))
        
        # Sample images
        sample_images = image_files[:sample_per_class]
        
        for img_path in sample_images:
            try:
                # Load image
                img = cv2.imread(str(img_path))
                if img is not None:
                    height, width, channels = img.shape
                    file_size = img_path.stat().st_size / 1024  # KB
                    
                    image_info.append({
                        'class': class_name,
                        'width': width,
                        'height': height,
                        'channels': channels,
                        'file_size_kb': file_size,
                        'aspect_ratio': width / height,
                        'total_pixels': width * height
                    })
            except Exception as e:
                print(f"⚠️ Error processing {img_path}: {e}")
    
    if not image_info:
        print("❌ No images could be analyzed")
        return None
    
    # Create DataFrame
    df_images = pd.DataFrame(image_info)
    
    print(f"\n📊 Image Statistics (from {len(df_images)} samples):")
    print(f"  Average Width: {df_images['width'].mean():.0f} px")
    print(f"  Average Height: {df_images['height'].mean():.0f} px")
    print(f"  Average File Size: {df_images['file_size_kb'].mean():.1f} KB")
    print(f"  Average Aspect Ratio: {df_images['aspect_ratio'].mean():.2f}")
    
    print(f"\n📏 Size Distribution:")
    print(f"  Width - Min: {df_images['width'].min()}, Max: {df_images['width'].max()}")
    print(f"  Height - Min: {df_images['height'].min()}, Max: {df_images['height'].max()}")
    
    # Check for consistency
    unique_sizes = df_images[['width', 'height']].drop_duplicates()
    print(f"\n🔍 Size Consistency:")
    print(f"  Unique size combinations: {len(unique_sizes)}")
    
    if len(unique_sizes) == 1:
        print(f"  ✅ All images have consistent size")
    else:
        print(f"  ⚠️  Images have varying sizes - normalization needed")
    
    return df_images

def detect_data_quality_issues(dataset_path):
    """Detect potential data quality issues"""
    
    print(f"\n🔍 DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    issues = []
    
    # Check for empty directories
    for split in ['train', 'validation', 'test']:
        split_path = dataset_path / split
        if split_path.exists():
            for class_dir in split_path.iterdir():
                if class_dir.is_dir():
                    image_count = 0
                    for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
                        image_count += len(list(class_dir.glob(ext)))
                    
                    if image_count == 0:
                        issues.append(f"Empty class directory: {split}/{class_dir.name}")
                    elif image_count < 10:
                        issues.append(f"Very few images ({image_count}) in: {split}/{class_dir.name}")
    
    # Check for broken symlinks (if using symlinks)
    if 'symlinks' in str(dataset_path):
        print("🔗 Checking symlink integrity...")
        broken_links = 0
        
        for split in ['train', 'validation', 'test']:
            split_path = dataset_path / split
            if split_path.exists():
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir():
                        for img_file in class_dir.iterdir():
                            if img_file.is_symlink() and not img_file.exists():
                                broken_links += 1
        
        if broken_links > 0:
            issues.append(f"Found {broken_links} broken symlinks")
        else:
            print("  ✅ All symlinks are valid")
    
    # Report issues
    if issues:
        print(f"\n⚠️  Found {len(issues)} potential issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✅ No major data quality issues detected")
    
    return issues

def create_visualizations(df_classes, df_images=None):
    """Create comprehensive visualizations"""
    
    print(f"\n📈 CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Set up the plotting
    plt.style.use('seaborn-v0_8')
    
    # 1. Class distribution plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Top 15 classes
    top_classes = df_classes.head(15)
    
    # Bar plot of class distribution
    axes[0, 0].bar(range(len(top_classes)), top_classes['total'])
    axes[0, 0].set_title('Top 15 Classes by Image Count', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Classes')
    axes[0, 0].set_ylabel('Number of Images')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Set x-axis labels
    class_labels = [name.replace('___', '\n') for name in top_classes.index]
    axes[0, 0].set_xticks(range(len(top_classes)))
    axes[0, 0].set_xticklabels(class_labels, rotation=45, ha='right')
    
    # Distribution histogram
    axes[0, 1].hist(df_classes['total'], bins=20, alpha=0.7, color='skyblue')
    axes[0, 1].set_title('Distribution of Images per Class', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Number of Images')
    axes[0, 1].set_ylabel('Number of Classes')
    
    # Split distribution
    split_data = df_classes[['train', 'validation', 'test']].sum()
    axes[1, 0].pie(split_data.values, labels=split_data.index, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')
    
    # Class balance visualization
    axes[1, 1].boxplot(df_classes['total'])
    axes[1, 1].set_title('Class Distribution Box Plot', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Number of Images')
    
    plt.tight_layout()
    plt.savefig('/Users/debabratapattnayak/web-dev/greencast/features/class_distribution_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Image properties visualization (if available)
    if df_images is not None and len(df_images) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Image size distribution
        axes[0, 0].scatter(df_images['width'], df_images['height'], alpha=0.6)
        axes[0, 0].set_title('Image Size Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Height (pixels)')
        
        # File size distribution
        axes[0, 1].hist(df_images['file_size_kb'], bins=20, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('File Size Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('File Size (KB)')
        axes[0, 1].set_ylabel('Number of Images')
        
        # Aspect ratio distribution
        axes[1, 0].hist(df_images['aspect_ratio'], bins=20, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Aspect Ratio Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Aspect Ratio (Width/Height)')
        axes[1, 0].set_ylabel('Number of Images')
        
        # Size by class (top 10 classes)
        top_classes_img = df_images['class'].value_counts().head(10).index
        df_top = df_images[df_images['class'].isin(top_classes_img)]
        
        sns.boxplot(data=df_top, x='class', y='file_size_kb', ax=axes[1, 1])
        axes[1, 1].set_title('File Size by Class (Top 10)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('File Size (KB)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('/Users/debabratapattnayak/web-dev/greencast/features/image_properties_analysis.png', 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    print("✅ Visualizations saved to features/ directory")

def main():
    """Main EDA function"""
    
    print("🚀 COMPREHENSIVE EDA AND DATA CLEANING")
    print("=" * 80)
    
    # 1. Analyze dataset structure
    dataset_info, dataset_path = analyze_dataset_structure()
    
    if dataset_info is None:
        return
    
    # 2. Class distribution analysis
    df_classes = create_class_distribution_analysis(dataset_info)
    
    # 3. Image properties analysis
    df_images = analyze_image_properties(dataset_path, sample_size=200)
    
    # 4. Data quality check
    issues = detect_data_quality_issues(dataset_path)
    
    # 5. Create visualizations
    create_visualizations(df_classes, df_images)
    
    # 6. Save analysis results
    results = {
        'dataset_info': dataset_info,
        'class_distribution': df_classes.to_dict(),
        'data_quality_issues': issues,
        'summary': {
            'total_classes': len(df_classes),
            'total_images': df_classes['total'].sum(),
            'imbalance_ratio': df_classes['total'].max() / df_classes['total'].min(),
            'avg_images_per_class': df_classes['total'].mean()
        }
    }
    
    # Save to JSON
    with open('/Users/debabratapattnayak/web-dev/greencast/features/eda_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save class distribution to CSV
    df_classes.to_csv('/Users/debabratapattnayak/web-dev/greencast/features/class_distribution.csv')
    
    if df_images is not None:
        df_images.to_csv('/Users/debabratapattnayak/web-dev/greencast/features/image_properties.csv', index=False)
    
    print(f"\n✅ EDA COMPLETE!")
    print(f"📊 Results saved to features/ directory")
    print(f"📈 Visualizations created and saved")
    print(f"📋 Summary: {len(df_classes)} classes, {df_classes['total'].sum():,} images")

if __name__ == "__main__":
    main()
