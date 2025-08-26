import os
def find_mmproj_pair(path):
    root = os.path.dirname(path)
    tenc_fname = os.path.basename(path).lower()
    target = []
    for fname in os.listdir(root):
        fname_lower = fname.lower()
        if fname_lower.endswith('.gguf') and 'mmproj' in fname_lower:
            target.append(fname)
    if not target:
        print(f"No mmproj file found for '{tenc_fname}'.")
        return None
    if len(target) > 1:
        print(f"Multiple mmproj files found for '{tenc_fname}', using first match.")
    mmproj_file = os.path.join(root, target[0])
    print(f"Using mmproj '{target[0]}' for '{tenc_fname}'.")
    return mmproj_file