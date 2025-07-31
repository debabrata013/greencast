#!/usr/bin/env python3
"""
Lightweight feature extractor that minimizes disk space usage
"""

import numpy as np
import cv2
from PIL import Image
from skimage import filters, exposure
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import label, regionprops
import tensorflow as tf
from tensorflow.keras.applications import VGG16  # Smaller than EfficientNet
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import warnings
warnings.filterwarnings('ignore')

class LightweightFeatureExtractor:
    """
    Memory and storage efficient feature extractor
    Uses only essential models to minimize disk usage
    """
    
    def __init__(self, use_deep_features=True):
        self.use_deep_features = use_deep_features
        self.models = {}
        
        if self.use_deep_features:
            try:
                print("Loading lightweight VGG16 model...")
                # Use only VGG16 (smaller than ResNet50 and EfficientNet)
                self.models['vgg16'] = VGG16(
                    weights='imagenet',
                    include_top=False,
                    pooling='avg',
                    input_shape=(224, 224, 3)
                )
                print("✓ VGG16 model loaded successfully")
            except Exception as e:
                print(f"⚠️ Could not load VGG16 due to storage constraints: {e}")
                print("Falling back to traditional features only")
                self.use_deep_features = False
    
    def extract_color_features(self, image):
        """Extract basic color statistics"""
        features = []
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # RGB statistics
        for channel in range(3):
            channel_data = image[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data)
            ])
        
        # HSV statistics (just hue and saturation)
        for channel in range(2):
            channel_data = hsv[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data)
            ])
        
        return features
    
    def extract_texture_features(self, image, distances=[1, 2], angles=[0, 45]):
        """Extract essential texture features using GLCM"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Reduce image size to save computation and memory
        gray_small = cv2.resize(gray, (128, 128))
        gray_norm = (gray_small / gray_small.max() * 255).astype(np.uint8)
        
        features = []
        
        # GLCM features (reduced set)
        for distance in distances:
            for angle in angles:
                try:
                    glcm = graycomatrix(
                        gray_norm, [distance], [np.radians(angle)], 
                        levels=64,  # Reduced from 256 to save memory
                        symmetric=True, normed=True
                    )
                    
                    # Extract only essential GLCM properties
                    features.extend([
                        graycoprops(glcm, 'contrast')[0, 0],
                        graycoprops(glcm, 'homogeneity')[0, 0],
                        graycoprops(glcm, 'energy')[0, 0]
                    ])
                except Exception as e:
                    print(f"Warning: GLCM computation failed: {e}")
                    features.extend([0, 0, 0])  # Default values
        
        # LBP features (simplified)
        try:
            lbp = local_binary_pattern(gray_small, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-7)  # Normalize
            features.extend(lbp_hist[:5])  # Use only first 5 bins
        except Exception as e:
            print(f"Warning: LBP computation failed: {e}")
            features.extend([0] * 5)
        
        return features
    
    def extract_shape_features(self, image):
        """Extract basic shape features"""
        # Convert to grayscale and threshold
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Simple thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Basic shape features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            features.extend([area / (image.shape[0] * image.shape[1]), circularity])
        else:
            features.extend([0, 0])
        
        return features
    
    def extract_deep_features(self, image):
        """Extract deep features using VGG16 if available"""
        if not self.use_deep_features or 'vgg16' not in self.models:
            return []
        
        try:
            # Preprocess image
            img_resized = cv2.resize(image, (224, 224))
            img_array = img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Extract features
            features = self.models['vgg16'].predict(img_array, verbose=0)
            return features.flatten()[:100]  # Use only first 100 features to save space
        
        except Exception as e:
            print(f"Warning: Deep feature extraction failed: {e}")
            return []
    
    def extract_all_features(self, image_path):
        """Extract all available features from an image"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to save memory
            image = cv2.resize(image, (256, 256))
            
            all_features = []
            
            # Extract different types of features
            color_features = self.extract_color_features(image)
            texture_features = self.extract_texture_features(image)
            shape_features = self.extract_shape_features(image)
            
            all_features.extend(color_features)
            all_features.extend(texture_features)
            all_features.extend(shape_features)
            
            # Add deep features if available
            if self.use_deep_features:
                deep_features = self.extract_deep_features(image)
                all_features.extend(deep_features)
            
            return np.array(all_features)
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # Return zeros if processing fails
            base_features = 50  # Color + texture + shape features
            deep_features = 100 if self.use_deep_features else 0
            return np.zeros(base_features + deep_features)

# Test the lightweight extractor
if __name__ == "__main__":
    print("Testing Lightweight Feature Extractor...")
    
    # Test without deep features first (no disk space needed)
    extractor_basic = LightweightFeatureExtractor(use_deep_features=False)
    print("✓ Basic feature extractor created")
    
    # Test with deep features (will fallback if no space)
    extractor_full = LightweightFeatureExtractor(use_deep_features=True)
    print("✓ Full feature extractor created")
    
    print("\nFeature extractor is ready to use!")
