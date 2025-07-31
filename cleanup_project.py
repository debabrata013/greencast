#!/usr/bin/env python3
"""
Project cleanup script - Remove unnecessary files and organize structure
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """Clean up unnecessary files and organize project structure"""
    
    base_path = Path("/Users/debabratapattnayak/web-dev/greencast")
    
    print("🧹 CLEANING UP PROJECT")
    print("=" * 50)
    
    # Files to remove (temporary/test files)
    files_to_remove = [
        "corrected_imports.py",
        "test_skimage_fix.py", 
        "test_imports.py",
        "test_all_imports.py",
        "test_notebook_fix.py",
        "test_directory_creation.py",
        "verify_mkdir_fix.py",
        "complete_solution.py",
        "complete_solution_fixed.py",
        "disk_space_solutions.py",
        "fix_notebook.py",
        "fix_syntax_error.py",
        "manual_cell_code.py",
        "notebook_replacement.py",
        "symlink_organizer.py",
        ".DS_Store"
    ]
    
    # Remove unnecessary Python files
    removed_files = []
    for file_name in files_to_remove:
        file_path = base_path / file_name
        if file_path.exists():
            file_path.unlink()
            removed_files.append(file_name)
            print(f"🗑️  Removed: {file_name}")
    
    # Remove __pycache__ directory
    pycache_path = base_path / "__pycache__"
    if pycache_path.exists():
        shutil.rmtree(pycache_path)
        print("🗑️  Removed: __pycache__/")
    
    # Clean up processed_data directory
    processed_data_path = base_path / "processed_data"
    
    # Remove unnecessary processed data directories
    dirs_to_remove = [
        "train",  # Empty directory
        "validation",  # Empty directory  
        "test",  # Empty directory
        ".qodo",  # Unnecessary
        "plantvillage_processed"  # Old version
    ]
    
    for dir_name in dirs_to_remove:
        dir_path = processed_data_path / dir_name
        if dir_path.exists():
            if dir_path.is_dir():
                shutil.rmtree(dir_path)
                print(f"🗑️  Removed directory: processed_data/{dir_name}")
    
    # Remove .DS_Store files
    for ds_store in processed_data_path.rglob(".DS_Store"):
        ds_store.unlink()
        print(f"🗑️  Removed: {ds_store.relative_to(base_path)}")
    
    # Remove backup notebook
    backup_notebook = base_path / "notebooks" / "04_feature_extraction.ipynb.backup"
    if backup_notebook.exists():
        backup_notebook.unlink()
        print("🗑️  Removed: notebooks/04_feature_extraction.ipynb.backup")
    
    print(f"\n✅ Cleanup complete! Removed {len(removed_files)} files")
    
    return removed_files

def organize_structure():
    """Organize the final project structure"""
    
    base_path = Path("/Users/debabratapattnayak/web-dev/greencast")
    
    print("\n📁 ORGANIZING PROJECT STRUCTURE")
    print("=" * 50)
    
    # Ensure required directories exist
    required_dirs = [
        "notebooks",
        "processed_data", 
        "models",
        "features",
        "src"
    ]
    
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Ensured directory exists: {dir_name}/")
    
    # Check what we have in processed_data
    processed_data_path = base_path / "processed_data"
    print(f"\n📊 Processed data contents:")
    for item in processed_data_path.iterdir():
        if item.is_dir():
            count = len(list(item.rglob("*")))
            print(f"  📁 {item.name}/ ({count} items)")
        else:
            size = item.stat().st_size / (1024*1024)  # MB
            print(f"  📄 {item.name} ({size:.1f} MB)")

if __name__ == "__main__":
    removed_files = cleanup_project()
    organize_structure()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Update all notebooks with lightweight feature extractor")
    print("2. Remove original dataset to save space") 
    print("3. Perform EDA on processed data")
    print("4. Clean and balance the dataset")
