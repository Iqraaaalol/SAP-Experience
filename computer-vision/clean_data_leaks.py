import os
import hashlib
import shutil
import config

# Use the active dataset from config
BASE = config.DATA_DIR
print(f"Cleaning dataset: {BASE}")
train_dir = os.path.join(BASE, 'train')
val_dir = os.path.join(BASE, 'val')
test_dir = os.path.join(BASE, 'test')

# Create backup directory
backup_dir = os.path.join(BASE, 'removed_duplicates_backup')
os.makedirs(backup_dir, exist_ok=True)

def list_files(root):
    files = []
    if not os.path.exists(root):
        return {}
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            full = os.path.join(dirpath, f)
            rel = os.path.relpath(full, root)
            files.append((rel.replace('\\', '/'), full))
    return dict(files)

def md5(path, block=65536):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block), b''):
            h.update(chunk)
    return h.hexdigest()

def clean_duplicates():
    print("\nStep 1: Loading file lists...")
    t = list_files(train_dir)
    v = list_files(val_dir)
    te = list_files(test_dir)
    
    print(f"Train files: {len(t)} | Val files: {len(v)} | Test files: {len(te)}")
    
    print("\nStep 2: Computing MD5 hashes...")
    # Build hash map for training data (keep these)
    train_hashes = {}
    for name, path in t.items():
        try:
            h = md5(path)
            train_hashes[h] = train_hashes.get(h, []) + [('train', name, path)]
        except Exception as e:
            print(f"Error hashing {path}: {e}")
    
    print(f"Unique training hashes: {len(train_hashes)}")
    
    # Check val and test files against training hashes
    files_to_remove = []
    
    print("\nStep 3: Checking validation set for duplicates...")
    for name, path in v.items():
        try:
            h = md5(path)
            if h in train_hashes:
                files_to_remove.append(('val', name, path, h))
        except Exception as e:
            print(f"Error hashing {path}: {e}")
    
    print("\nStep 4: Checking test set for duplicates...")
    for name, path in te.items():
        try:
            h = md5(path)
            if h in train_hashes:
                files_to_remove.append(('test', name, path, h))
        except Exception as e:
            print(f"Error hashing {path}: {e}")
    
    print(f"\n{'='*70}")
    print(f"DUPLICATES TO REMOVE: {len(files_to_remove)}")
    print(f"{'='*70}")
    
    if not files_to_remove:
        print("\n✓ No duplicates found! Dataset is clean.")
        return
    
    # Show summary
    val_count = sum(1 for split, _, _, _ in files_to_remove if split == 'val')
    test_count = sum(1 for split, _, _, _ in files_to_remove if split == 'test')
    print(f"  Val duplicates:  {val_count}")
    print(f"  Test duplicates: {test_count}")
    
    # Confirm before deletion
    print(f"\nFiles will be moved to: {backup_dir}")
    response = input("\nProceed with removal? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("Aborted.")
        return
    
    print("\nStep 5: Removing duplicate files...")
    removed_count = 0
    error_count = 0
    
    for split, name, path, h in files_to_remove:
        try:
            # Create backup structure
            backup_path = os.path.join(backup_dir, split, name)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Move file to backup
            shutil.move(path, backup_path)
            removed_count += 1
            
            if removed_count % 100 == 0:
                print(f"  Removed {removed_count}/{len(files_to_remove)}...")
                
        except Exception as e:
            print(f"  Error removing {path}: {e}")
            error_count += 1
    
    print(f"\n{'='*70}")
    print(f"CLEANUP COMPLETE")
    print(f"{'='*70}")
    print(f"Successfully removed: {removed_count}")
    print(f"Errors: {error_count}")
    print(f"Backup location: {backup_dir}")
    print(f"\nDataset is now clean. Re-run check_data_leak.py to verify.")

if __name__ == '__main__':
    clean_duplicates()
