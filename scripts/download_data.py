"""
Dataset Download Script for AuthNet
Downloads and organizes DTD + MVTec AD (texture categories) datasets.
"""

import os
import sys
import tarfile
import zipfile
import shutil
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest_path: str, desc: str = "Downloading"):
    """Download a file with progress bar."""
    if os.path.exists(dest_path):
        print(f"  [SKIP] {desc} — already exists at {dest_path}")
        return
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  [DOWNLOAD] {desc}")
    print(f"    URL: {url}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, dest_path, reporthook=t.update_to)
    
    print(f"  [DONE] Saved to {dest_path}")


def download_dtd():
    """Download and extract the Describable Textures Dataset (DTD)."""
    print("\n" + "=" * 60)
    print("STEP 1: Downloading Describable Textures Dataset (DTD)")
    print("=" * 60)
    
    dtd_url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    archive_path = os.path.join(config.DATA_DIR, "dtd-r1.0.1.tar.gz")
    
    # Check if already extracted
    dtd_images_dir = os.path.join(config.DTD_DIR, "images")
    if os.path.exists(dtd_images_dir) and len(os.listdir(dtd_images_dir)) >= 40:
        print("  [SKIP] DTD already extracted.")
        return
    
    # Download
    download_file(dtd_url, archive_path, desc="DTD (~600MB)")
    
    # Extract
    print("  [EXTRACT] Extracting DTD archive...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(config.DATA_DIR)
    
    # The archive extracts to data/dtd/ with images/ and labels/ subdirs
    extracted_dir = os.path.join(config.DATA_DIR, "dtd")
    if os.path.exists(extracted_dir) and extracted_dir != config.DTD_DIR:
        # Move if needed
        if not os.path.exists(config.DTD_DIR):
            shutil.move(extracted_dir, config.DTD_DIR)
    
    # Clean up archive
    if os.path.exists(archive_path):
        os.remove(archive_path)
    
    # Verify
    if os.path.exists(dtd_images_dir):
        categories = os.listdir(dtd_images_dir)
        total_images = sum(
            len(os.listdir(os.path.join(dtd_images_dir, cat)))
            for cat in categories if os.path.isdir(os.path.join(dtd_images_dir, cat))
        )
        print(f"  [OK] DTD: {len(categories)} texture categories, {total_images} images")
    else:
        print("  [WARN] DTD extraction may have failed. Check data/dtd/images/")


def download_mvtec():
    """
    Download MVTec AD dataset (texture categories only).
    MVTec requires accepting license, so we provide instructions for manual download
    and also attempt direct download of the full dataset.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Setting up MVTec AD (Texture Categories)")
    print("=" * 60)
    
    # Check if already present
    existing_cats = []
    for cat in config.MVTEC_CATEGORIES:
        cat_dir = os.path.join(config.MVTEC_DIR, cat)
        if os.path.exists(cat_dir) and os.path.exists(os.path.join(cat_dir, "train")):
            existing_cats.append(cat)
    
    if len(existing_cats) == len(config.MVTEC_CATEGORIES):
        print("  [SKIP] All MVTec categories already present.")
        return
    
    # MVTec AD download URL (public, direct link)
    mvtec_url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
    archive_path = os.path.join(config.DATA_DIR, "mvtec_anomaly_detection.tar.xz")
    
    # Try to download
    if not os.path.exists(archive_path):
        print("  [INFO] Attempting to download MVTec AD (~3.5GB)...")
        print("  [INFO] If download fails, manually download from:")
        print("         https://www.mvtec.com/company/research/datasets/mvtec-ad")
        print(f"         and extract to: {config.MVTEC_DIR}")
        
        try:
            download_file(mvtec_url, archive_path, desc="MVTec AD (~3.5GB)")
        except Exception as e:
            print(f"\n  [ERROR] Download failed: {e}")
            print("  [INFO] Please download MVTec AD manually:")
            print("         1. Go to https://www.mvtec.com/company/research/datasets/mvtec-ad")
            print("         2. Download the dataset")
            print(f"         3. Extract texture categories to: {config.MVTEC_DIR}")
            print(f"         4. Expected structure: {config.MVTEC_DIR}/leather/train/good/")
            _create_mvtec_placeholder()
            return
    
    # Extract only texture categories
    print("  [EXTRACT] Extracting MVTec texture categories...")
    try:
        with tarfile.open(archive_path, "r:xz") as tar:
            members = []
            for member in tar.getmembers():
                # Only extract our texture categories
                for cat in config.MVTEC_CATEGORIES:
                    if member.name.startswith(cat + "/") or member.name == cat:
                        members.append(member)
                        break
            
            print(f"  [INFO] Extracting {len(members)} files for categories: {config.MVTEC_CATEGORIES}")
            tar.extractall(config.MVTEC_DIR, members=members)
    except Exception as e:
        print(f"  [ERROR] Extraction failed: {e}")
        print("  [INFO] Trying full extraction...")
        try:
            with tarfile.open(archive_path, "r:xz") as tar:
                tar.extractall(config.MVTEC_DIR)
        except Exception as e2:
            print(f"  [ERROR] Full extraction also failed: {e2}")
            _create_mvtec_placeholder()
            return
    
    # Clean up archive (it's large)
    if os.path.exists(archive_path):
        print("  [CLEANUP] Removing archive to save space...")
        os.remove(archive_path)
    
    # Verify
    _verify_mvtec()


def _create_mvtec_placeholder():
    """Create placeholder structure for MVTec if download fails."""
    print("\n  [INFO] Creating placeholder structure for MVTec AD...")
    print("  [INFO] You can still train on DTD alone, or add MVTec data later.\n")
    
    for cat in config.MVTEC_CATEGORIES:
        for split in ["train/good", "test/good", "test/defect"]:
            os.makedirs(os.path.join(config.MVTEC_DIR, cat, split), exist_ok=True)


def _verify_mvtec():
    """Verify MVTec dataset structure."""
    print("\n  MVTec AD Dataset Summary:")
    for cat in config.MVTEC_CATEGORIES:
        cat_dir = os.path.join(config.MVTEC_DIR, cat)
        if os.path.exists(cat_dir):
            train_good = os.path.join(cat_dir, "train", "good")
            test_good = os.path.join(cat_dir, "test", "good")
            
            n_train = len(os.listdir(train_good)) if os.path.exists(train_good) else 0
            n_test_good = len(os.listdir(test_good)) if os.path.exists(test_good) else 0
            
            # Count defect types
            test_dir = os.path.join(cat_dir, "test")
            defect_types = []
            n_test_defect = 0
            if os.path.exists(test_dir):
                for d in os.listdir(test_dir):
                    if d != "good" and os.path.isdir(os.path.join(test_dir, d)):
                        defect_types.append(d)
                        n_test_defect += len(os.listdir(os.path.join(test_dir, d)))
            
            print(f"    {cat:>10s}: {n_train} train (good), "
                  f"{n_test_good} test (good), {n_test_defect} test (defect)")
            if defect_types:
                print(f"               defect types: {', '.join(defect_types)}")
        else:
            print(f"    {cat:>10s}: [NOT FOUND]")


def create_combined_dataset():
    """
    Create a unified combined dataset from DTD + MVTec for training.
    Organizes images into class-based subdirectories with train/test splits.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Creating Combined Dataset")
    print("=" * 60)
    
    train_dir = os.path.join(config.COMBINED_DIR, "train")
    test_dir = os.path.join(config.COMBINED_DIR, "test")
    
    # Check if already created
    if os.path.exists(train_dir) and len(os.listdir(train_dir)) > 5:
        print("  [SKIP] Combined dataset already exists.")
        return
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    class_idx = 0
    
    # --- Add DTD classes ---
    dtd_images_dir = os.path.join(config.DTD_DIR, "images")
    dtd_labels_dir = os.path.join(config.DTD_DIR, "labels")
    
    if os.path.exists(dtd_images_dir):
        print("  [DTD] Processing DTD texture categories...")
        
        # Use DTD's official train/test splits if available
        dtd_train_files = {}
        dtd_test_files = {}
        
        # Read split files
        for split_name, split_dict in [("train1.txt", dtd_train_files), 
                                         ("test1.txt", dtd_test_files)]:
            split_file = os.path.join(dtd_labels_dir, split_name)
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split("/")
                            cat = parts[0]
                            if cat not in split_dict:
                                split_dict[cat] = []
                            split_dict[cat].append(line)
        
        # Also add val to train
        val_file = os.path.join(dtd_labels_dir, "val1.txt")
        if os.path.exists(val_file):
            with open(val_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split("/")
                        cat = parts[0]
                        if cat not in dtd_train_files:
                            dtd_train_files[cat] = []
                        dtd_train_files[cat].append(line)
        
        categories = sorted(os.listdir(dtd_images_dir))
        for cat in categories:
            cat_path = os.path.join(dtd_images_dir, cat)
            if not os.path.isdir(cat_path):
                continue
            
            class_name = f"dtd_{cat}"
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
            
            # Use official splits if available
            if cat in dtd_train_files and cat in dtd_test_files:
                for rel_path in dtd_train_files[cat]:
                    src = os.path.join(dtd_images_dir, rel_path)
                    if os.path.exists(src):
                        dst = os.path.join(train_dir, class_name, os.path.basename(rel_path))
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                
                for rel_path in dtd_test_files[cat]:
                    src = os.path.join(dtd_images_dir, rel_path)
                    if os.path.exists(src):
                        dst = os.path.join(test_dir, class_name, os.path.basename(rel_path))
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
            else:
                # Fallback: 80/20 split
                images = sorted(os.listdir(cat_path))
                images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
                split_idx = int(len(images) * config.TRAIN_SPLIT)
                
                for img in images[:split_idx]:
                    src = os.path.join(cat_path, img)
                    dst = os.path.join(train_dir, class_name, img)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                
                for img in images[split_idx:]:
                    src = os.path.join(cat_path, img)
                    dst = os.path.join(test_dir, class_name, img)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
            
            class_idx += 1
        
        print(f"  [DTD] Added {class_idx} texture classes")
    else:
        print("  [WARN] DTD images not found. Skipping DTD.")
    
    # --- Add MVTec classes ---
    mvtec_classes = 0
    for cat in config.MVTEC_CATEGORIES:
        cat_dir = os.path.join(config.MVTEC_DIR, cat)
        train_good = os.path.join(cat_dir, "train", "good")
        test_good = os.path.join(cat_dir, "test", "good")
        
        if not os.path.exists(train_good):
            continue
        
        # Add "good" (authentic) as a class
        class_name = f"mvtec_{cat}_good"
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
        
        for img in os.listdir(train_good):
            if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                src = os.path.join(train_good, img)
                dst = os.path.join(train_dir, class_name, img)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        
        if os.path.exists(test_good):
            for img in os.listdir(test_good):
                if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    src = os.path.join(test_good, img)
                    dst = os.path.join(test_dir, class_name, img)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
        
        mvtec_classes += 1
        
        # Add defect types as separate classes (for triplet training)
        test_dir_cat = os.path.join(cat_dir, "test")
        if os.path.exists(test_dir_cat):
            for defect_type in os.listdir(test_dir_cat):
                defect_path = os.path.join(test_dir_cat, defect_type)
                if defect_type == "good" or not os.path.isdir(defect_path):
                    continue
                
                defect_class_name = f"mvtec_{cat}_{defect_type}"
                os.makedirs(os.path.join(train_dir, defect_class_name), exist_ok=True)
                os.makedirs(os.path.join(test_dir, defect_class_name), exist_ok=True)
                
                defect_images = [f for f in os.listdir(defect_path)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                # Split defect images: 60% train, 40% test
                split_idx = max(1, int(len(defect_images) * 0.6))
                
                for img in defect_images[:split_idx]:
                    src = os.path.join(defect_path, img)
                    dst = os.path.join(train_dir, defect_class_name, img)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                
                for img in defect_images[split_idx:]:
                    src = os.path.join(defect_path, img)
                    dst = os.path.join(test_dir, defect_class_name, img)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                
                mvtec_classes += 1
    
    if mvtec_classes > 0:
        print(f"  [MVTec] Added {mvtec_classes} classes (good + defect types)")
    else:
        print("  [WARN] No MVTec data found. Using DTD only.")
    
    # --- Summary ---
    print("\n  Combined Dataset Summary:")
    train_classes = [d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))]
    test_classes = [d for d in os.listdir(test_dir) 
                    if os.path.isdir(os.path.join(test_dir, d))]
    
    # Remove empty classes
    for split, split_dir in [("train", train_dir), ("test", test_dir)]:
        for cls in os.listdir(split_dir):
            cls_path = os.path.join(split_dir, cls)
            if os.path.isdir(cls_path) and len(os.listdir(cls_path)) == 0:
                os.rmdir(cls_path)
    
    # Recount
    train_classes = [d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))]
    
    total_train = sum(len(os.listdir(os.path.join(train_dir, c))) for c in train_classes)
    total_test = sum(len(os.listdir(os.path.join(test_dir, c))) 
                     for c in os.listdir(test_dir) 
                     if os.path.isdir(os.path.join(test_dir, c)))
    
    print(f"    Train: {len(train_classes)} classes, {total_train} images")
    print(f"    Test:  {len([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])} classes, {total_test} images")
    
    # Show samples per class distribution
    samples_per_class = [len(os.listdir(os.path.join(train_dir, c))) for c in train_classes]
    if samples_per_class:
        print(f"    Samples/class: min={min(samples_per_class)}, "
              f"max={max(samples_per_class)}, "
              f"mean={sum(samples_per_class)/len(samples_per_class):.1f}")


def main():
    print("=" * 60)
    print("  AuthNet Dataset Setup")
    print("=" * 60)
    print(f"  Data directory: {config.DATA_DIR}")
    
    # Step 1: Download DTD
    download_dtd()
    
    # Step 2: Download/setup MVTec AD
    download_mvtec()
    
    # Step 3: Create combined dataset
    create_combined_dataset()
    
    print("\n" + "=" * 60)
    print("  Setup Complete!")
    print("=" * 60)
    print(f"\n  Combined dataset: {config.COMBINED_DIR}")
    print("  You can now run: python -m src.train")


if __name__ == "__main__":
    main()
