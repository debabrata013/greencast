#!/usr/bin/env python3
"""
Update all notebooks to use the lightweight feature extractor and fix imports
"""

import json
import re
from pathlib import Path

def update_notebook_imports(notebook_path):
    """Update imports in a notebook to use corrected versions"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    updated = False
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source_lines = cell['source']
            new_source = []
            
            for line in source_lines:
                # Fix greycomatrix/greycoprops imports
                if 'greycomatrix' in line or 'greycoprops' in line:
                    line = line.replace('greycomatrix', 'graycomatrix')
                    line = line.replace('greycoprops', 'graycoprops')
                    updated = True
                
                # Fix mkdir calls
                if '.mkdir(exist_ok=True)' in line and 'parents=True' not in line:
                    line = line.replace('.mkdir(exist_ok=True)', '.mkdir(parents=True, exist_ok=True)')
                    updated = True
                
                new_source.append(line)
            
            cell['source'] = new_source
    
    if updated:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        return True
    
    return False

def add_lightweight_extractor_to_notebooks():
    """Add the lightweight feature extractor to relevant notebooks"""
    
    notebooks_dir = Path("/Users/debabratapattnayak/web-dev/greencast/notebooks")
    
    # Lightweight extractor code to add
    extractor_code = [
        "# LIGHTWEIGHT FEATURE EXTRACTOR - OPTIMIZED FOR DISK SPACE\n",
        "import sys\n",
        "sys.path.append('/Users/debabratapattnayak/web-dev/greencast')\n",
        "from lightweight_feature_extractor import LightweightFeatureExtractor\n",
        "\n",
        "# Enhanced DeepFeatureExtractor using lightweight backend\n",
        "class DeepFeatureExtractor:\n",
        "    def __init__(self):\n",
        "        print(\"Initializing lightweight feature extractor...\")\n",
        "        try:\n",
        "            self.extractor = LightweightFeatureExtractor(use_deep_features=True)\n",
        "            print(\"✅ Deep feature extractor initialized with VGG16!\")\n",
        "        except Exception as e:\n",
        "            print(f\"⚠️ Deep features failed: {e}\")\n",
        "            self.extractor = LightweightFeatureExtractor(use_deep_features=False)\n",
        "            print(\"✅ Basic feature extractor initialized!\")\n",
        "    \n",
        "    def extract_all_features(self, image_path):\n",
        "        return self.extractor.extract_all_features(image_path)\n",
        "    \n",
        "    def extract_vgg16_features(self, image_path):\n",
        "        import cv2\n",
        "        image = cv2.imread(str(image_path))\n",
        "        if image is not None:\n",
        "            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n",
        "            return self.extractor.extract_deep_features(image)\n",
        "        return []\n",
        "    \n",
        "    def extract_resnet50_features(self, image_path):\n",
        "        return self.extract_vgg16_features(image_path)\n",
        "    \n",
        "    def extract_efficientnet_features(self, image_path):\n",
        "        return self.extract_vgg16_features(image_path)\n"
    ]
    
    # Update 04_feature_extraction.ipynb
    feature_notebook = notebooks_dir / "04_feature_extraction.ipynb"
    if feature_notebook.exists():
        with open(feature_notebook, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Find and replace the DeepFeatureExtractor class
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'class DeepFeatureExtractor:' in source or 'DeepFeatureExtractor' in source:
                    cell['source'] = extractor_code
                    break
        
        with open(feature_notebook, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        
        print("✅ Updated 04_feature_extraction.ipynb")

def update_all_notebooks():
    """Update all notebooks in the project"""
    
    notebooks_dir = Path("/Users/debabratapattnayak/web-dev/greencast/notebooks")
    
    print("📝 UPDATING ALL NOTEBOOKS")
    print("=" * 50)
    
    for notebook_path in notebooks_dir.glob("*.ipynb"):
        print(f"\n🔧 Processing: {notebook_path.name}")
        
        # Update imports and fix common issues
        if update_notebook_imports(notebook_path):
            print(f"  ✅ Fixed imports and mkdir calls")
        else:
            print(f"  ℹ️  No updates needed")
    
    # Add lightweight extractor to feature extraction notebook
    add_lightweight_extractor_to_notebooks()
    
    print("\n✅ All notebooks updated!")

if __name__ == "__main__":
    update_all_notebooks()
