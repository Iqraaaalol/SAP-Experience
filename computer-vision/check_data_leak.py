import os
import hashlib
import config

# Use the active dataset from config instead of hardcoding
BASE = config.DATA_DIR
print(f"Checking dataset: {BASE}")
train_dir = os.path.join(BASE, 'train')
val_dir = os.path.join(BASE, 'val')
test_dir = os.path.join(BASE, 'test')

def list_files(root):
    files = []
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


def find_overlaps():
    t = list_files(train_dir)
    v = list_files(val_dir)
    te = list_files(test_dir)

    names_train = set(t.keys())
    names_val = set(v.keys())
    names_test = set(te.keys())

    print(f"Train files: {len(names_train)} | Val files: {len(names_val)} | Test files: {len(names_test)}")

    name_overlap_train_val = names_train & names_val
    name_overlap_train_test = names_train & names_test
    name_overlap_val_test = names_val & names_test

    print("\nFilename overlaps:")
    print(f"  train∩val: {len(name_overlap_train_val)}")
    print(f"  train∩test: {len(name_overlap_train_test)}")
    print(f"  val∩test: {len(name_overlap_val_test)}")

    if name_overlap_train_val:
        for n in list(name_overlap_train_val)[:20]:
            print("   ", n)

    # Now check content duplicates by MD5 across splits only
    print("\nComputing MD5 hashes for files (this may take a moment)...")
    md5_map = {}
    
    # Tag each file with its split
    for name, path in t.items():
        try:
            h = md5(path)
            md5_map.setdefault(h, []).append(('train', name, path))
        except Exception as e:
            pass
    
    for name, path in v.items():
        try:
            h = md5(path)
            md5_map.setdefault(h, []).append(('val', name, path))
        except Exception as e:
            pass
    
    for name, path in te.items():
        try:
            h = md5(path)
            md5_map.setdefault(h, []).append(('test', name, path))
        except Exception as e:
            pass
    
    # Filter to only cross-split duplicates
    cross_split_duplicates = {}
    for h, items in md5_map.items():
        if len(items) > 1:
            # Get unique splits in this duplicate group
            splits = set(item[0] for item in items)
            if len(splits) > 1:  # Only keep if files appear in different splits
                cross_split_duplicates[h] = items
    
    print(f"\n{'='*70}")
    print(f"CROSS-SPLIT DUPLICATES (CRITICAL DATA LEAK)")
    print(f"{'='*70}")
    print(f"Total duplicate groups crossing splits: {len(cross_split_duplicates)}")
    
    if cross_split_duplicates:
        total_leaked = sum(len(items) for items in cross_split_duplicates.values())
        print(f"Total leaked files: {total_leaked}")
        print(f"\nShowing first 20 groups:\n")
        
        for h, items in list(cross_split_duplicates.items())[:20]:
            # Count splits
            split_counts = {}
            for split, name, path in items:
                split_counts[split] = split_counts.get(split, 0) + 1
            
            split_summary = ', '.join([f"{split}: {count}" for split, count in sorted(split_counts.items())])
            print(f"Hash: {h} [{split_summary}]")
            for split, name, path in items:
                print(f"  [{split:5s}] {name}")
            print()
    else:
        print("\n✓ No cross-split duplicates found! Dataset splits are clean.")

if __name__ == '__main__':
    find_overlaps()
