import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
def quantize_to_fp8(tensor):
    if tensor.dtype != torch.bfloat16:
        raise ValueError('Input tensor must be in BF16 format.')
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = torch.max(torch.abs(tensor)) / fp8_max
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized_tensor = (tensor / scale).to(torch.float8_e4m3fn)
    return quantized_tensor, scale
import os
safetensors_files = [file for file in os.listdir() if file.endswith(
    '.safetensors')]
if safetensors_files:
    print('Safetensors file(s) available. Select which one to convert:')
    for index, file_name in enumerate(safetensors_files, start=1):
        print(f'{index}. {file_name}')
    choice = input(f'Enter your choice (1 to {len(safetensors_files)}): ')
    try:
        choice_index = int(choice) - 1
        selected_file = safetensors_files[choice_index]
        print(f'Model file: {selected_file} is selected!')
        input_file = selected_file
        output_file = (
            f'{os.path.splitext(input_file)[0]}_fp8_e4m3fn.safetensors')
        data = load_file(input_file)
        quantized_data = {}
        print('Starting quantization process...')
        for key, tensor in tqdm(data.items(), desc='Quantizing tensors',
            unit='tensor'):
            if key.endswith('.weight') and tensor.ndim >= 2:
                tensor = tensor.to(dtype=torch.bfloat16, device='cuda')
                quantized_tensor, scale = quantize_to_fp8(tensor)
                layer = key[:-len('.weight')]
                quantized_data[key] = quantized_tensor.cpu()
                quantized_data[layer + '.scale_weight'] = scale.to(torch.bfloat16).reshape(1).cpu()
            else:
                quantized_data[key] = tensor
        quantized_data['scaled_fp8'] = torch.ones(2, dtype=torch.float8_e4m3fn)
        save_file(quantized_data, output_file)
        print(f'Quantized safetensors saved to {output_file}.')
    except (ValueError, IndexError):
        print('Invalid choice. Please enter a valid number.')
else:
    print('No safetensors files are available in the current directory.')
    input('--- Press ENTER To Exit ---')