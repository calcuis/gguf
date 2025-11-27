import comfy.sd
import comfy.ops
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import torch, numpy, os, json, logging, collections, nodes, folder_paths
from safetensors.torch import load_file, save_file
from typing import Dict, Tuple
from tqdm import tqdm as loading
from .gguf_connector import reader as gr
from .gguf_connector.writer import GGUFWriter, GGMLQuantizationType
from .gguf_connector.const import GGML_QUANT_VERSION, LlamaFileType
from .gguf_connector.quant import quantize, dequantize, QuantError
from .gguf_connector.quant5a import dequantize_tensor, is_quantized, is_torch_compatible
from .gguf_connector.mmj import find_mmproj_pair, find_tokenzier_pair
from .gguf_connector.tkn import get_field, tokenizer_builder
pig = os.path.join(os.path.dirname(__file__), 'version.json')
with open(pig, 'r') as file:
    data = json.load(file)
arrays = {}
for key, value in data[0].items():
    arrays[key] = value
class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False
    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)
        try:
            from comfy.lora import calculate_weight
        except Exception:
            calculate_weight = self.calculate_weight
        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = load_patch_to_device(patches, self.load_device if
                self.patch_on_device else self.offload_device)
            out_weight.patches = [(calculate_weight, patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', [
                    'weight', 'inplace_update'])(weight.to(device=self.
                    offload_device, copy=inplace_update), inplace_update)
            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight,
                    device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)
            out_weight = calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)
        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)
    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, 'patches', [])
                if len(patches) > 0:
                    p.patches = []
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)
    mmap_released = False
    def load(self, *args, force_patch_weights=False, **kwargs):
        super().load(*args, force_patch_weights=True, **kwargs)
        if not self.mmap_released:
            linked = []
            if kwargs.get('lowvram_model_memory', 0) > 0:
                for n, m in self.model.named_modules():
                    if hasattr(m, 'weight'):
                        device = getattr(m.weight, 'device', None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, 'bias'):
                        device = getattr(m.bias, 'device', None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked:
                print(f'Attempting to release mmap ({len(linked)})')
                for n, m in linked:
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True
    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        n.patch_on_device = getattr(self, 'patch_on_device', False)
        return n
class GGMLTensor(torch.Tensor):
    def __init__(self, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        self.patches = patches
    def __new__(cls, *args, tensor_type, tensor_shape, patches=[], **kwargs):
        return super().__new__(cls, *args, **kwargs)
    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, 'tensor_type', None)
        new.tensor_shape = getattr(self, 'tensor_shape', new.data.shape)
        new.patches = getattr(self, 'patches', []).copy()
        return new
    def clone(self, *args, **kwargs):
        return self
    def detach(self, *args, **kwargs):
        return self
    def copy_(self, *args, **kwargs):
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            print(f"ignoring 'copy_' on tensor: {e}")
    def empty_(self, size, *args, **kwargs):
        new_tensor = super().empty_(size, *args, **kwargs)
        return GGMLTensor(new_tensor, tensor_type=getattr(self,
            'tensor_type', None), tensor_shape=size, patches=getattr(self,
            'patches', []).copy())
    @property
    def shape(self):
        if not hasattr(self, 'tensor_shape'):
            self.tensor_shape = self.size()
        return self.tensor_shape
if hasattr(torch, 'compiler') and hasattr(torch.compiler, 'disable'):
    torch_compiler_disable = torch.compiler.disable
else:
    def torch_compiler_disable(*args, **kwargs):
        def noop(x):
            return x
        return noop
class GGMLLayer(torch.nn.Module):
    comfy_cast_weights = True
    dequant_dtype = None
    patch_dtype = None
    largest_layer = False
    torch_compatible_tensor_types = {None, gr.GGMLQuantizationType.F32, gr.
        GGMLQuantizationType.F16}
    def is_ggml_quantized(self, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight
        if bias is None:
            bias = self.bias
        return is_quantized(weight) or is_quantized(bias)
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight, bias = state_dict.get(f'{prefix}weight'), state_dict.get(
            f'{prefix}bias')
        if self.is_ggml_quantized(weight=weight, bias=bias) or isinstance(self,
            torch.nn.Linear):
            return self.ggml_load_from_state_dict(state_dict, prefix, *args,
                **kwargs)
        return super()._load_from_state_dict(state_dict, prefix, *args, **
            kwargs)
    def ggml_load_from_state_dict(self, state_dict, prefix, local_metadata,
        strict, missing_keys, unexpected_keys, error_msgs):
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            if k[prefix_len:] == 'weight':
                self.weight = torch.nn.Parameter(v, requires_grad=False)
            elif k[prefix_len:] == 'bias' and v is not None:
                self.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                missing_keys.append(k)
        if self.weight is None and isinstance(self, torch.nn.Linear):
            v = torch.zeros(self.in_features, self.out_features)
            self.weight = torch.nn.Parameter(v, requires_grad=False)
            missing_keys.append(prefix + 'weight')
        if getattr(self.weight, 'is_largest_weight', False):
            self.largest_layer = True
    def _save_to_state_dict(self, *args, **kwargs):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(*args, **kwargs)
        return super()._save_to_state_dict(*args, **kwargs)
    def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
        weight = torch.zeros_like(self.weight, device=torch.device('meta'))
        destination[prefix + 'weight'] = weight
        if self.bias is not None:
            bias = torch.zeros_like(self.bias, device=torch.device('meta'))
            destination[prefix + 'bias'] = bias
        if self.largest_layer:
            shape = getattr(self.weight, 'tensor_shape', self.weight.shape)
            dtype = torch.float16 if self.dequant_dtype == 'target' or self.dequant_dtype is None else self.dequant_dtype
            temp = torch.empty(*shape, device=torch.device('meta'), dtype=dtype)
            destination[prefix + 'temp.weight'] = temp
        return
        destination[prefix + 'weight'] = self.get_weight(self.weight)
        if bias is not None:
            destination[prefix + 'bias'] = self.get_weight(self.bias)
    def get_weight(self, tensor, dtype):
        if tensor is None:
            return
        patch_list = []
        device = tensor.device
        for function, patches, key in getattr(tensor, 'patches', []):
            patch_list += load_patch_to_device(patches, device)
        weight = dequantize_tensor(tensor, dtype, self.dequant_dtype)
        if isinstance(weight, GGMLTensor):
            weight = torch.Tensor(weight)
        if patch_list:
            if self.patch_dtype is None:
                weight = function(patch_list, weight, key)
            else:
                patch_dtype = (dtype if self.patch_dtype == 'target' else
                    self.patch_dtype)
                weight = function(patch_list, weight, key, patch_dtype)
        return weight
    @torch_compiler_disable()
    def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype
        =None):
        if input is not None:
            if dtype is None:
                dtype = getattr(input, 'dtype', torch.float32)
            if bias_dtype is None:
                bias_dtype = dtype
            if device is None:
                device = input.device
        bias = None
        non_blocking = comfy.model_management.device_supports_non_blocking(
            device)
        if s.bias is not None:
            bias = s.get_weight(s.bias.to(device), dtype)
            bias = comfy.ops.cast_to(bias, bias_dtype, device, non_blocking=non_blocking, copy=False)
        weight = s.get_weight(s.weight.to(device), dtype)
        weight = comfy.ops.cast_to(weight, dtype, device, non_blocking=
            non_blocking, copy=False)
        return weight, bias
    def forward_comfy_cast_weights(self, input, *args, **kwargs):
        if self.is_ggml_quantized():
            out = self.forward_ggml_cast_weights(input, *args, **kwargs)
        else:
            out = super().forward_comfy_cast_weights(input, *args, **kwargs)
        if isinstance(out, GGMLTensor):
            out = torch.Tensor(out)
        return out
    def forward_ggml_cast_weights(self, input):
        raise NotImplementedError
class GGMLOps(comfy.ops.manual_cast):
    class Linear(GGMLLayer, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=
            None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.linear(input, weight, bias)
    class Conv2d(GGMLLayer, comfy.ops.manual_cast.Conv2d):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return self._conv_forward(input, weight, bias)
    class Embedding(GGMLLayer, comfy.ops.manual_cast.Embedding):
        def forward_ggml_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if (self.weight.dtype == torch.float16 or self.weight.dtype ==
                torch.bfloat16):
                out_dtype = None
            weight, _bias = self.cast_bias_weight(self, device=input.device,
                dtype=out_dtype)
            return torch.nn.functional.embedding(input, weight, self.
                padding_idx, self.max_norm, self.norm_type, self.
                scale_grad_by_freq, self.sparse).to(dtype=output_dtype)
    class LayerNorm(GGMLLayer, comfy.ops.manual_cast.LayerNorm):
        def forward_ggml_cast_weights(self, input):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.layer_norm(input, self.
                normalized_shape, weight, bias, self.eps)
    class GroupNorm(GGMLLayer, comfy.ops.manual_cast.GroupNorm):
        def forward_ggml_cast_weights(self, input):
            weight, bias = self.cast_bias_weight(input)
            return torch.nn.functional.group_norm(input, self.num_groups,
                weight, bias, self.eps)
def load_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, tuple):
        return tuple(load_patch_to_device(x, device) for x in item)
    elif isinstance(item, list):
        return [load_patch_to_device(x, device) for x in item]
    else:
        return item
def get_folder_names_and_paths(key, targets=[]):
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    target = next((x for x in targets if x in folder_paths.
        folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = orig or base, {'.gguf'}
    if base and base != orig:
        logging.warning(
            f'Unknown file list already present on key {key}: {base}')
get_folder_names_and_paths('model_gguf', ['diffusion_models', 'unet'])
get_folder_names_and_paths('clip_gguf', ['text_encoders', 'clip'])
get_folder_names_and_paths('encoder_gguf', ['audio_encoders'])
get_folder_names_and_paths('vae_gguf', ['vae'])
def get_orig_shape(reader, tensor_name):
    field_key = f'comfy.gguf.orig_shape.{tensor_name}'
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) != 2 or field.types[0
        ] != gr.GGUFValueType.ARRAY or field.types[1
        ] != gr.GGUFValueType.INT32:
        raise TypeError(f'Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}')
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in
        field.data))
def load_gguf_sd(path, handle_prefix='model.diffusion_model.', return_arch=
    False):
    reader = gr.GGUFReader(path)
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)
    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))
    compat = None
    arch_str = get_field(reader, 'general.architecture', str)
    if arch_str is None:
        compat = 'sd.cpp'
    elif arch_str not in arrays['PIG_ARCH_LIST'] and arch_str not in arrays['TXT_ARCH_LIST']:
        raise ValueError(f"Unknown architecture: {arch_str!r}")
    state_dict, qtype_dict = {}, {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name
        torch_tensor = torch.from_numpy(tensor.data)
        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            if compat == 'sd.cpp' and arch_str == 'sdxl':
                if any([tensor_name.endswith(x) for x in ('.proj_in.weight',
                    '.proj_out.weight')]):
                    while len(shape) > 2 and shape[-1] == 1:
                        shape = shape[:-1]
        if tensor.tensor_type in {gr.GGMLQuantizationType.F32, gr.
            GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
        state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.
            tensor_type, tensor_shape=shape)
        tensor_type_str = getattr(tensor.tensor_type, 'name', repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1
    print('gguf qtypes: ' + ', '.join(f'{k} ({v})' for k, v in qtype_dict.items()))
    qsd = {k: v for k, v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True
    if return_arch:
        return state_dict, arch_str
    return state_dict
def tensor_swap(raw_sd, key_map):
    sd = {}
    for k, v in raw_sd.items():
        for s, d in key_map.items():
            k = k.replace(s, d)
        sd[k] = v
    return sd
def pig_work(raw_sd):
    sd = {}
    for k, v in raw_sd.items():
        sd[k] = v
    return sd
def llama_permute(raw_sd, n_head, n_head_kv):
    sd = {}
    permute = lambda x, h: x.reshape(h, x.shape[0] // h // 2, 2, *x.shape[1:]
        ).swapaxes(1, 2).reshape(x.shape)
    for k, v in raw_sd.items():
        if k.endswith(('q_proj.weight', 'q_proj.bias')):
            v.data = permute(v.data, n_head)
        if k.endswith(('k_proj.weight', 'k_proj.bias')):
            v.data = permute(v.data, n_head_kv)
        sd[k] = v
    return sd
def handle_visual_tensor(path):
    vsd = load_gguf_sd(path)
    if "v.patch_embd.weight.1" in vsd:
        w1 = dequantize_tensor(vsd.pop("v.patch_embd.weight"), dtype=torch.float32)
        w2 = dequantize_tensor(vsd.pop("v.patch_embd.weight.1"), dtype=torch.float32)
        vsd["v.patch_embd.weight"] = torch.stack([w1, w2], dim=2)
    vsd = tensor_swap(vsd, arrays['V7'])
    if "visual.blocks.0.attn_q.weight" in vsd:
        attns = {}
        for k,v in vsd.items():
            if any(x in k for x in ["attn_q", "attn_k", "attn_v"]):
                k_attn, k_name = k.rsplit(".attn_", 1)
                k_attn += ".attn.qkv." + k_name.split(".")[-1]
                if k_attn not in attns:
                    attns[k_attn] = {}
                attns[k_attn][k_name] = dequantize_tensor(v, dtype=(torch.bfloat16 if is_quantized(v) else torch.float16))
        for k,v in attns.items():
            suffix = k.split(".")[-1]
            vsd[k] = torch.cat([v[f"q.{suffix}"],v[f"k.{suffix}"],v[f"v.{suffix}"]], dim=0)
        del attns
    return vsd
def load_gguf_mmproj(path):
    target = find_mmproj_pair(path)
    if not target:
        vsd = {}
    else:
        vsd = handle_visual_tensor(target)
    return vsd
def load_safetensors_tokenizer(path):
    target = find_tokenzier_pair(path)
    if not target:
        tsd = {}
    else:
        tsd = comfy.utils.load_torch_file(target, safe_load=True)
    return tsd
def load_gguf_clip(path):
    sd, arch = load_gguf_sd(path, return_arch=True)
    if arch in {'t5', 't5encoder'}:
        temb_key = 'token_embd.weight'
        if temb_key in sd and (sd[temb_key].shape == (256384, 4096) or sd[temb_key].shape == (256384, 768)):
            sd['spiece_model'] = tokenizer_builder(path)
            sd = tensor_swap(sd, arrays['T5'])
        elif temb_key in sd and sd[temb_key].shape == (32128, 768):
            sd = tensor_swap(sd, arrays['B5'])
        else:
            sd = tensor_swap(sd, arrays['T5'])
    elif arch in {'llama', "qwen2vl", "qwen3", "dog"}:
        sd = tensor_swap(sd, arrays['L3'])
        if arch == "llama":
            sd = llama_permute(sd, 32, 8)
        if arch == "qwen2vl":
            vsd = load_gguf_mmproj(path)
            sd.update(vsd)
        if arch == "dog":
            vsd = handle_visual_tensor(path)
            sd.update(vsd)
    elif arch in {'gemma2'}:
        sd["spiece_model"] = tokenizer_builder(path)
        sd = tensor_swap(sd, arrays['G2'])
    elif arch in {'cat'}:
        sd = pig_work(sd)
        tsd = load_safetensors_tokenizer(path)
        sd.update(tsd)
    elif arch in {'cow'}:
        sd = pig_work(sd)
        spiece_key = 'tekken_model'
        if spiece_key in sd:sd["tekken_model"] = sd["tekken_model"].to(torch.uint8)
        else:sd["spiece_model"] = sd["spiece_model"].to(torch.uint8)
    elif arch in {'pig'}:
        sd = pig_work(sd)
    else:
        pass
    return sd
class LoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        gguf_names = [x for x in folder_paths.get_filename_list('model_gguf')]
        return {'required': {'gguf_name': (gguf_names,)}}
    RETURN_TYPES = 'MODEL',
    FUNCTION = 'load_model'
    CATEGORY = 'gguf'
    TITLE = 'GGUF Loader'
    def load_model(self, gguf_name, dequant_dtype=None, patch_dtype=None,
        patch_on_device=None):
        ops = GGMLOps()
        if dequant_dtype in ('default', None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ['target']:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)
        if patch_dtype in ('default', None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ['target']:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)
        model_path = folder_paths.get_full_path('unet', gguf_name)
        sd = load_gguf_sd(model_path)
        model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=
            {'custom_operations': ops})
        if model is None:
            logging.error('ERROR UNSUPPORTED MODEL {}'.format(model_path))
            raise RuntimeError('ERROR: Could not detect model type of: {}'.
                format(model_path))
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device
        return model,
class LoaderGGUFAdvanced(LoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        model_names = [x for x in folder_paths.get_filename_list('model_gguf')]
        return {'required': {'gguf_name': (model_names,), 'dequant_dtype':
            (['default', 'target', 'float32', 'float16', 'bfloat16'], {
            'default': 'default'}), 'patch_dtype': (['default', 'target',
            'float32', 'float16', 'bfloat16'], {'default': 'default'}),
            'patch_on_device': ('BOOLEAN', {'default': False})}}
    TITLE = 'GGUF Loader (Advanced)'
def get_clip_type(type):
    clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
    return clip_type
def get_device(device):
    model_options = {}
    if device == "cpu":
        model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
    return model_options
class ClipLoaderGGUF:
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.CLIPLoader.INPUT_TYPES()
        return {'required': {'clip_name': (s.get_filename_list(),), 'type':
                             base['required']['type']},
                             'optional':{'device':(['default','cpu'],{'advanced':True}),}}
    RETURN_TYPES = 'CLIP',
    FUNCTION = 'load_clip'
    CATEGORY = 'gguf'
    TITLE = 'GGUF CLIP Loader'
    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list('clip')
        files += folder_paths.get_filename_list('clip_gguf')
        return sorted(files)
    def load_data(self, ckpt_paths):
        clip_data = []
        for p in ckpt_paths:
            if p.endswith('.gguf'):
                sd = load_gguf_clip(p)
            else:
                sd = comfy.utils.load_torch_file(p, safe_load=True)
            clip_data.append(sd)
        return clip_data
    def load_patcher(self, clip_paths, clip_type, clip_data):
        clip = comfy.sd.load_text_encoder_state_dicts(clip_type=clip_type,
            state_dicts=clip_data, model_options={'custom_operations':
            GGMLOps, 'initial_device': comfy.model_management.
            text_encoder_offload_device()}, embedding_directory=
            folder_paths.get_folder_paths('embeddings'))
        clip.patcher = GGUFModelPatcher.clone(clip.patcher)
        return clip
    def load_clip(self, clip_name, type='stable_diffusion', device='default'):
        clip_path = folder_paths.get_full_path('clip', clip_name)
        if clip_name.endswith('.safetensors'):
            clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=get_clip_type(type), model_options=get_device(device))
            return (clip,)
        else:
            return (self.load_patcher([clip_path], get_clip_type(type), self.load_data([clip_path])), get_device('default'))
class DualClipLoaderGGUF(ClipLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        base = nodes.DualCLIPLoader.INPUT_TYPES()
        file_options = s.get_filename_list(),
        return {'required': {'clip_name1':file_options, 'clip_name2':file_options, 'type':
                             base['required']['type']},
                             'optional':{'device':(['default','cpu'],{'advanced':True}),}}
    TITLE = 'GGUF DualCLIP Loader'
    def load_clip(self, clip_name1, clip_name2, type, device='default'):
        clip_path1 = folder_paths.get_full_path('clip', clip_name1)
        clip_path2 = folder_paths.get_full_path('clip', clip_name2)
        clip_paths = clip_path1, clip_path2
        if device != 'default' and (clip_name1.endswith('.safetensors') and clip_name2.endswith('.safetensors')):
            clip = comfy.sd.load_clip(ckpt_paths=clip_paths, embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=get_clip_type(type), model_options=get_device(device))
            return (clip,)
        else:
            return (self.load_patcher(clip_paths, get_clip_type(type), self.load_data(clip_paths)), get_device(device))
class TripleClipLoaderGGUF(ClipLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = s.get_filename_list(),
        return {'required': {'clip_name1': file_options, 'clip_name2':
            file_options, 'clip_name3': file_options}}
    TITLE = 'GGUF TripleCLIP Loader'
    def load_clip(self, clip_name1, clip_name2, clip_name3, type='sd3'):
        clip_path1 = folder_paths.get_full_path('clip', clip_name1)
        clip_path2 = folder_paths.get_full_path('clip', clip_name2)
        clip_path3 = folder_paths.get_full_path('clip', clip_name3)
        clip_paths = clip_path1, clip_path2, clip_path3
        return (self.load_patcher(clip_paths, get_clip_type(type), self.load_data(clip_paths)),)
class QuadrupleClipLoaderGGUF(ClipLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        file_options = s.get_filename_list(),
        return {'required': {'clip_name1': file_options, 'clip_name2':
            file_options, 'clip_name3': file_options, 'clip_name4': file_options}}
    TITLE = 'GGUF QuadrupleCLIP Loader'
    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4, type='hunyuan_video'):
        clip_path1 = folder_paths.get_full_path('clip', clip_name1)
        clip_path2 = folder_paths.get_full_path('clip', clip_name2)
        clip_path3 = folder_paths.get_full_path('clip', clip_name3)
        clip_path4 = folder_paths.get_full_path('clip', clip_name4)
        clip_paths = clip_path1, clip_path2, clip_path3, clip_path4
        return (self.load_patcher(clip_paths, get_clip_type(type), self.load_data(clip_paths)),)
QUANTIZATION_THRESHOLD = 1024
REARRANGE_THRESHOLD = 512
MAX_TENSOR_NAME_LENGTH = 127
class ModelTemplate:
    arch = 'invalid'
    keys_detect = []
    keys_banned = []
class ModelFlux(ModelTemplate):
    arch = 'flux'
    keys_detect = [('transformer_blocks.0.attn.norm_added_k.weight',), (
        'double_blocks.0.img_attn.proj.weight',)]
    keys_banned = ['transformer_blocks.0.attn.norm_added_k.weight']
class ModelSD3(ModelTemplate):
    arch = 'sd3'
    keys_detect = [('transformer_blocks.0.attn.add_q_proj.weight',), (
        'joint_blocks.0.x_block.attn.qkv.weight',)]
    keys_banned = ['transformer_blocks.0.attn.add_q_proj.weight']
class ModelSDXL(ModelTemplate):
    arch = 'sdxl'
    keys_detect = [('down_blocks.0.downsamplers.0.conv.weight',
        'add_embedding.linear_1.weight'), ('input_blocks.3.0.op.weight',
        'input_blocks.6.0.op.weight', 'output_blocks.2.2.conv.weight',
        'output_blocks.5.2.conv.weight'), ('label_emb.0.0.weight',)]
class ModelSD1(ModelTemplate):
    arch = 'sd1'
    keys_detect = [('down_blocks.0.downsamplers.0.conv.weight',), (
        'input_blocks.3.0.op.weight', 'input_blocks.6.0.op.weight',
        'input_blocks.9.0.op.weight', 'output_blocks.2.1.conv.weight',
        'output_blocks.5.2.conv.weight', 'output_blocks.8.2.conv.weight')]
arch_list = [ModelFlux, ModelSD3, ModelSDXL, ModelSD1]
def is_model_arch(model, state_dict):
    matched = False
    invalid = False
    for match_list in model.keys_detect:
        if all(key in state_dict for key in match_list):
            matched = True
            invalid = any(key in state_dict for key in model.keys_banned)
            break
    assert not invalid, 'Model architecture not supported for alpha (please opt to use zero)'
    return matched
def detect_arch(state_dict):
    model_arch = None
    for arch in arch_list:
        if is_model_arch(arch, state_dict):
            model_arch = arch
            break
    assert model_arch is not None, 'Unknown model architecture detected!'
    return model_arch
def load_state_dict(path):
    state_dict = load_file(path)
    prefix = None
    for pfx in ['model.diffusion_model.', 'model.', 'net.']:
        if any([x.startswith(pfx) for x in state_dict.keys()]):
            prefix = pfx
            break
    sd = {}
    for k, v in state_dict.items():
        if prefix and prefix not in k:
            continue
        if prefix:
            k = k.replace(prefix, '')
        sd[k] = v
    return sd
def load_model(path):
    state_dict = load_state_dict(path)
    model_arch = detect_arch(state_dict)
    print(f'* Architecture detected from input: {model_arch.arch}')
    writer = GGUFWriter(path=None, arch=model_arch.arch)
    return writer, state_dict, model_arch
def load_pig_state(path):
    pig_state = load_file(path)
    sd = {}
    for k, v in pig_state.items():
        sd[k] = v
    return sd
def load_pig(path):
    state_dict = load_pig_state(path)
    model_arch = arrays['TXT_ARCH_LIST'][0]
    writer = GGUFWriter(path=None, arch=model_arch)
    return writer, state_dict, model_arch
def handle_tensors(args, writer, state_dict, model_arch):
    name_lengths = tuple(sorted(((key, len(key)) for key in state_dict.keys
        ()), key=lambda item: item[1], reverse=True))
    if not name_lengths:
        return
    max_name_len = name_lengths[0][1]
    if max_name_len > MAX_TENSOR_NAME_LENGTH:
        bad_list = ', '.join(f'{key!r} ({namelen})' for key, namelen in
            name_lengths if namelen > MAX_TENSOR_NAME_LENGTH)
        raise ValueError(f'Can only handle tensor names up to {MAX_TENSOR_NAME_LENGTH} characters. Tensors exceeding the limit: {bad_list}')
    for key, data in loading(state_dict.items()):
        old_dtype = data.dtype
        if data.dtype == torch.bfloat16:
            data = data.to(torch.float32).numpy()
        elif data.dtype in [getattr(torch, 'float8_e4m3fn', '_invalid'),
            getattr(torch, 'float8_e5m2', '_invalid')]:
            data = data.to(torch.float16).numpy()
        else:
            data = data.numpy()
        n_dims = len(data.shape)
        data_shape = data.shape
        data_qtype = getattr(GGMLQuantizationType, 'BF16' if old_dtype ==
            torch.bfloat16 else 'F16')
        n_params = 1
        for dim_size in data_shape:
            n_params *= dim_size
        blacklist = {'time_embedding.', 'add_embedding.', 'time_in.',
            'txt_in.', 'vector_in.', 'img_in.', 'guidance_in.', 'final_layer.'}
        if old_dtype in (torch.float32, torch.bfloat16):
            if n_dims == 1:
                data_qtype = GGMLQuantizationType.F32
            elif n_params <= QUANTIZATION_THRESHOLD:
                data_qtype = GGMLQuantizationType.F32
            elif '.weight' in key and any(x in key for x in blacklist):
                data_qtype = GGMLQuantizationType.F32
        try:
            data = quantize(data, data_qtype)
        except (AttributeError, QuantError) as e:
            loading.write(f'falling back to F16: {e}')
            data_qtype = GGMLQuantizationType.F16
            data = quantize(data, data_qtype)
        new_name = key
        shape_str = f"{{{', '.join(str(n) for n in reversed(data.shape))}}}"
        loading.write(
            f"{f'%-{max_name_len + 4}s' % f'{new_name}'} {old_dtype} --> {data_qtype.name}, shape = {shape_str}"
            )
        writer.add_tensor(new_name, data, raw_dtype=data_qtype)
if 'select_safetensors' not in folder_paths.folder_names_and_paths:
    orig = folder_paths.folder_names_and_paths.get('diffusion_models',
        folder_paths.folder_names_and_paths.get('checkpoints', [[], set()]))
    folder_paths.folder_names_and_paths['select_safetensors'] = orig[0], {'.safetensors'}
class GGUFSave:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'select_safetensors': (s.get_filename_list(),)}}
    RETURN_TYPES = ()
    FUNCTION = 'save'
    OUTPUT_NODE = True
    CATEGORY = 'gguf'
    TITLE = 'GGUF Convertor (Alpha)'
    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list('select_safetensors')
        return sorted(files)
    def save(self, select_safetensors):
        path = folder_paths.get_full_path('select_safetensors',
            select_safetensors)
        writer, state_dict, model_arch = load_model(path)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        if next(iter(state_dict.values())).dtype == torch.bfloat16:
            output_path = (f'{self.output_dir}/{os.path.splitext(select_safetensors)[0]}-bf16.gguf')
            writer.add_file_type(LlamaFileType.MOSTLY_BF16)
        else:
            output_path = (f'{self.output_dir}/{os.path.splitext(select_safetensors)[0]}-f16.gguf')
            writer.add_file_type(LlamaFileType.MOSTLY_F16)
        if os.path.isfile(output_path):
            input('Output exists enter to continue or ctrl+c to abort!')
        handle_tensors(output_path, writer, state_dict, model_arch)
        writer.write_header_to_file(path=output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        return {}
class GGUFRun:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'select_safetensors': (s.get_filename_list(),)}}
    RETURN_TYPES = ()
    FUNCTION = 'run'
    OUTPUT_NODE = True
    CATEGORY = 'gguf'
    TITLE = 'GGUF Convertor (Zero)'
    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list('select_safetensors')
        return sorted(files)
    def run(self, select_safetensors):
        path = folder_paths.get_full_path('select_safetensors',
            select_safetensors)
        writer, state_dict, model_arch = load_pig(path)
        writer.add_quantization_version(GGML_QUANT_VERSION)
        if next(iter(state_dict.values())).dtype == torch.bfloat16:
            output_path = (f'{self.output_dir}/{os.path.splitext(select_safetensors)[0]}-bf16.gguf')
            writer.add_file_type(LlamaFileType.MOSTLY_BF16)
        else:
            output_path = (f'{self.output_dir}/{os.path.splitext(select_safetensors)[0]}-f16.gguf')
            writer.add_file_type(LlamaFileType.MOSTLY_F16)
        if os.path.isfile(output_path):
            input('Output exists enter to continue or ctrl+c to abort!')
        handle_tensors(output_path, writer, state_dict, model_arch)
        writer.write_header_to_file(path=output_path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        return {}
def quantize_to_fp8(tensor):
    if tensor.dtype != torch.bfloat16:
        raise ValueError('Input tensor must be in BF16 format.')
    tensor = tensor.to(torch.float16)
    fp8_max = 240.0
    fp8_min = -fp8_max
    clamped_tensor = tensor.clamp(min=fp8_min, max=fp8_max)
    scale = fp8_max / torch.max(torch.abs(clamped_tensor))
    quantized_tensor = torch.round(clamped_tensor * scale) / scale
    return quantized_tensor.to(torch.float8_e4m3fn)
class TENSORCut:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'select_safetensors': (s.get_filename_list(),)}}
    RETURN_TYPES = ()
    FUNCTION = 'cut'
    OUTPUT_NODE = True
    CATEGORY = 'gguf'
    TITLE = 'TENSOR Cutter (Beta)'
    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list('select_safetensors')
        return sorted(files)
    def cut(self, select_safetensors):
        input_file = folder_paths.get_full_path('select_safetensors', select_safetensors)
        output_file = (f'{self.output_dir}/{os.path.splitext(select_safetensors)[0]}_fp8_e4m3fn.safetensors')
        data = load_file(input_file)
        quantized_data = {}
        print('Starting quantization process...')
        for key, tensor in loading(data.items(), desc='Quantizing tensors', unit='tensor'):
            tensor = tensor.to(dtype=torch.bfloat16, device='cuda')
            quantized_tensor = quantize_to_fp8(tensor)
            quantized_data[key] = quantized_tensor.cpu()
        save_file(quantized_data, output_file)
        print(f'Quantized safetensors saved to {output_file}.')
        return {}
class TENSORBoost:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'select_safetensors': (s.get_filename_list(),)}}
    RETURN_TYPES = ()
    FUNCTION = 'boost'
    OUTPUT_NODE = True
    CATEGORY = 'gguf'
    TITLE = 'TENSOR Booster'
    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list('select_safetensors')
        return sorted(files)
    def boost(self, select_safetensors):
        input_file = folder_paths.get_full_path('select_safetensors',
            select_safetensors)
        output_file = (f'{self.output_dir}/{os.path.splitext(select_safetensors)[0]}_fp32.safetensors')
        data = load_file(input_file)
        quantized_data = {}
        print('Starting quantization process...')
        for key, tensor in loading(data.items(), desc="Converting tensors", unit="tensor"):
            quantized_data[key] = tensor.to(torch.float32)
        save_file(quantized_data, output_file)
        print(f'Quantized safetensors saved to {output_file}.')
        return {}
def load_gguf_and_extract_metadata(gguf_path):
    reader = gr.GGUFReader(gguf_path)
    tensors_metadata = []
    for tensor in reader.tensors:
        tensor_metadata = {'name': tensor.name, 'shape': tuple(tensor.shape
            .tolist()), 'n_elements': tensor.n_elements, 'n_bytes': tensor.
            n_bytes, 'data_offset': tensor.data_offset, 'type': tensor.
            tensor_type}
        tensors_metadata.append(tensor_metadata)
    return reader, tensors_metadata
def convert_gguf_to_safetensors(gguf_path, output_path, use_bf16):
    reader, tensors_metadata = load_gguf_and_extract_metadata(gguf_path)
    print(f'Extracted {len(tensors_metadata)} tensors from GGUF file')
    tensors_dict: dict[str, torch.Tensor] = {}
    for i, tensor_info in enumerate(loading(tensors_metadata, desc=
        'Converting tensors', unit='tensor')):
        tensor_name = tensor_info['name']
        tensor_data = reader.get_tensor(i)
        weights = dequantize(tensor_data.data, tensor_data.tensor_type).copy()
        try:
            if use_bf16:
                weights_tensor = torch.from_numpy(weights).to(dtype=torch.float32)
                weights_tensor = weights_tensor.to(torch.bfloat16)
            else:
                weights_tensor = torch.from_numpy(weights).to(dtype=torch.float16)
            weights_hf = weights_tensor
        except Exception as e:
            print(f"Error during BF16 conversion for tensor '{tensor_name}': {e}")
            weights_tensor = torch.from_numpy(weights.astype(numpy.float32)).to(torch.float16)
            weights_hf = weights_tensor
        tensors_dict[tensor_name] = weights_hf
    metadata = {key: str(reader.get_field(key)) for key in reader.fields}
    save_file(tensors_dict, output_path, metadata=metadata)
    print('Conversion complete!')
if 'select_gguf' not in folder_paths.folder_names_and_paths:
    orig = folder_paths.folder_names_and_paths.get('diffusion_models',
        folder_paths.folder_names_and_paths.get('unet', [[], set()]))
    folder_paths.folder_names_and_paths['select_gguf'] = orig[0], {'.gguf'}
class GGUFUndo:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'select_gguf': (s.get_filename_list(),)}}
    RETURN_TYPES = ()
    FUNCTION = 'undo'
    OUTPUT_NODE = True
    CATEGORY = 'gguf'
    TITLE = 'GGUF Convertor (Reverse)'
    @classmethod
    def get_filename_list(s):
        files = []
        files += folder_paths.get_filename_list('select_gguf')
        return sorted(files)
    def undo(self, select_gguf):
        in_file = folder_paths.get_full_path('select_gguf', select_gguf)
        out_file = (f'{self.output_dir}/{os.path.splitext(select_gguf)[0]}_fp16.safetensors')
        use_bf16 = False
        convert_gguf_to_safetensors(in_file, out_file, use_bf16)
        return {}
class VaeGGUF:
    @staticmethod
    def vae_list():
        vaes = []
        vaes += folder_paths.get_filename_list('vae')
        vaes += folder_paths.get_filename_list('vae_gguf')
        approx_vaes = folder_paths.get_filename_list('vae_approx')
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False
        for v in approx_vaes:
            if v.startswith('taesd_decoder.'):
                sd1_taesd_dec = True
            elif v.startswith('taesd_encoder.'):
                sd1_taesd_enc = True
            elif v.startswith('taesdxl_decoder.'):
                sdxl_taesd_dec = True
            elif v.startswith('taesdxl_encoder.'):
                sdxl_taesd_enc = True
            elif v.startswith('taesd3_decoder.'):
                sd3_taesd_dec = True
            elif v.startswith('taesd3_encoder.'):
                sd3_taesd_enc = True
            elif v.startswith('taef1_encoder.'):
                f1_taesd_dec = True
            elif v.startswith('taef1_decoder.'):
                f1_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append('taesd')
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append('taesdxl')
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append('taesd3')
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append('taef1')
        return vaes
    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list('vae_approx')
        encoder = next(filter(lambda a: a.startswith('{}_encoder.'.format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith('{}_decoder.'.format(name)), approx_vaes))
        enc = comfy.utils.load_torch_file(folder_paths.
            get_full_path_or_raise('vae_approx', encoder))
        for k in enc:
            sd['taesd_encoder.{}'.format(k)] = enc[k]
        dec = comfy.utils.load_torch_file(folder_paths.
            get_full_path_or_raise('vae_approx', decoder))
        for k in dec:
            sd['taesd_decoder.{}'.format(k)] = dec[k]
        if name == 'taesd':
            sd['vae_scale'] = torch.tensor(0.18215)
            sd['vae_shift'] = torch.tensor(0.0)
        elif name == 'taesdxl':
            sd['vae_scale'] = torch.tensor(0.13025)
            sd['vae_shift'] = torch.tensor(0.0)
        elif name == 'taesd3':
            sd['vae_scale'] = torch.tensor(1.5305)
            sd['vae_shift'] = torch.tensor(0.0609)
        elif name == 'taef1':
            sd['vae_scale'] = torch.tensor(0.3611)
            sd['vae_shift'] = torch.tensor(0.1159)
        return sd
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'vae_name': (s.vae_list(),)}}
    RETURN_TYPES = 'VAE',
    FUNCTION = 'load_vae'
    CATEGORY = 'gguf'
    TITLE = 'GGUF VAE Loader'
    def load_vae(self, vae_name):
        if vae_name.endswith('.gguf'):
            vae_path = folder_paths.get_full_path_or_raise('vae_gguf', vae_name)
            sd = load_gguf_clip(vae_path)
        elif vae_name in ['taesd', 'taesdxl', 'taesd3', 'taef1']:
            sd = self.load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path_or_raise('vae', vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=sd)
        return vae,
class AudioEncoderLoaderGGUF:
    @staticmethod
    def get_encoder_list():
        encoders = []
        encoders += folder_paths.get_filename_list('audio_encoders')
        encoders += folder_paths.get_filename_list('encoder_gguf')
        return sorted(encoders)
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio_encoder_name": (s.get_encoder_list(), ),}}
    RETURN_TYPES = ("AUDIO_ENCODER",)
    FUNCTION = "load_model"
    CATEGORY = 'gguf'
    TITLE = 'GGUF AudioEncoder Loader'
    def load_model(self, audio_encoder_name):
        if audio_encoder_name.endswith('.gguf'):
            encoder_path = folder_paths.get_full_path_or_raise('encoder_gguf', audio_encoder_name)
            sd = load_gguf_clip(encoder_path)
        else:
            audio_encoder_name = folder_paths.get_full_path_or_raise("audio_encoders", audio_encoder_name)
            sd = comfy.utils.load_torch_file(audio_encoder_name, safe_load=True)
        from comfy.audio_encoders.audio_encoders import load_audio_encoder_from_sd
        audio_encoder = load_audio_encoder_from_sd(sd)
        # audio_encoder = comfy.audio_encoders.audio_encoders.load_audio_encoder_from_sd(sd)
        if audio_encoder is None:
            raise RuntimeError("ERROR: audio encoder file is invalid and does not contain a valid model.")
        return audio_encoder,
NODE_CLASS_MAPPINGS = {
    "LoaderGGUF": LoaderGGUF,
    "ClipLoaderGGUF": ClipLoaderGGUF,
    "DualClipLoaderGGUF": DualClipLoaderGGUF,
    "TripleClipLoaderGGUF": TripleClipLoaderGGUF,
    "QuadrupleClipLoaderGGUF": QuadrupleClipLoaderGGUF,
    "AudioEncoderLoaderGGUF": AudioEncoderLoaderGGUF,
    "LoaderGGUFAdvanced": LoaderGGUFAdvanced,
    "VaeGGUF": VaeGGUF,
    "GGUFUndo": GGUFUndo,
    "GGUFSave": GGUFSave,
    "GGUFRun": GGUFRun,
    "TENSORCut": TENSORCut,
    "TENSORBoost": TENSORBoost,
}
