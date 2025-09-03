import os
def find_mmproj_pair(path):
    root = os.path.dirname(path)
    txt_fname = os.path.basename(path).lower()
    target = []
    for fname in os.listdir(root):
        fname_lower = fname.lower()
        if fname_lower.endswith('.gguf') and 'mmproj' in fname_lower:
            target.append(fname)
    if not target:
        print(f"No mmproj file found for '{txt_fname}'.")
        return None
    if len(target) > 1:
        print(f"Multiple mmproj files found for '{txt_fname}', using first match.")
    mmproj_file = os.path.join(root, target[0])
    print(f"Using mmproj '{target[0]}' for '{txt_fname}'.")
    return mmproj_file
def find_tokenzier_pair(path):
    root = os.path.dirname(path)
    txt_fname = os.path.basename(path).lower()
    target = []
    for fname in os.listdir(root):
        fname_lower = fname.lower()
        if fname_lower.endswith('.safetensors') and 'tokenizer' in fname_lower:
            target.append(fname)
    if not target:
        print(f"No tokenizer file found for '{txt_fname}'.")
        return None
    if len(target) > 1:
        print(f"Multiple tokenizer files found for '{txt_fname}', using first match.")
    tokenizer_file = os.path.join(root, target[0])
    print(f"Using tokenizer '{target[0]}' for '{txt_fname}'.")
    return tokenizer_file