from __future__ import annotations
import logging, os, sys, shutil, struct, tempfile
from dataclasses import dataclass
from enum import Enum, auto
from math import prod
from pathlib import Path
from io import BufferedWriter
from typing import IO, Any, Sequence, Mapping
from string import ascii_letters, digits
import numpy as np
from .const import GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION, GGMLQuantizationType, GGUFEndian, GGUFValueType, Keys, RopeScalingType, PoolingType, TokenType, ExpertGatingFuncType
from .quant import quant_shape_from_byte_shape
logger = logging.getLogger(__name__)
SHARD_NAME_FORMAT = '{:s}-{:05d}-of-{:05d}.gguf'
@dataclass
class TensorInfo:
    shape: Sequence[int]
    dtype: GGMLQuantizationType
    nbytes: int
    tensor: np.ndarray[Any, Any] | None = None
@dataclass
class GGUFValue:
    value: Any
    type: GGUFValueType
    sub_type: GGUFValueType | None = None
class WriterState(Enum):
    NO_FILE = auto()
    EMPTY = auto()
    HEADER = auto()
    KV_DATA = auto()
    TI_DATA = auto()
    WEIGHTS = auto()
class GGUFWriter:
    fout: list[BufferedWriter] | None
    path: Path | None
    temp_file: tempfile.SpooledTemporaryFile[bytes] | None
    tensors: list[dict[str, TensorInfo]]
    kv_data: list[dict[str, GGUFValue]]
    state: WriterState
    _simple_value_packing = {GGUFValueType.UINT8: 'B', GGUFValueType.INT8:
        'b', GGUFValueType.UINT16: 'H', GGUFValueType.INT16: 'h',
        GGUFValueType.UINT32: 'I', GGUFValueType.INT32: 'i', GGUFValueType.
        FLOAT32: 'f', GGUFValueType.UINT64: 'Q', GGUFValueType.INT64: 'q',
        GGUFValueType.FLOAT64: 'd', GGUFValueType.BOOL: '?'}
    def __init__(self, path, arch, use_temp_file=False, endianess=
        GGUFEndian.LITTLE, split_max_tensors=0, split_max_size=0, dry_run=
        False, small_first_shard=False):
        self.fout = None
        self.path = Path(path) if path else None
        self.arch = arch
        self.endianess = endianess
        self.data_alignment = GGUF_DEFAULT_ALIGNMENT
        self.use_temp_file = use_temp_file
        self.temp_file = None
        self.tensors = [{}]
        self.kv_data = [{}]
        self.split_max_tensors = split_max_tensors
        self.split_max_size = split_max_size
        self.dry_run = dry_run
        self.small_first_shard = small_first_shard
        logger.info('gguf: This GGUF file is for {0} Endian only'.format(
            'Big' if self.endianess == GGUFEndian.BIG else 'Little'))
        self.state = WriterState.NO_FILE
        if self.small_first_shard:
            self.tensors.append({})
        self.add_architecture()
    def get_total_parameter_count(self):
        total_params = 0
        shared_params = 0
        expert_params = 0
        expert_sum = 0
        n_expert_tensors = 0
        last_lora_a: tuple[str, TensorInfo] | None = None
        for tensors in self.tensors:
            for name, info in tensors.items():
                shape = info.shape
                if name.endswith('.lora_a'):
                    last_lora_a = name, info
                    continue
                elif name.endswith('.lora_b'):
                    if last_lora_a is None or last_lora_a[0] != name[:-1
                        ] + 'a':
                        logger.warning(
                            "can't measure LoRA size correctly, tensor order is unusual"
                            )
                        return 0, 0, 0, 0
                    else:
                        shape = *shape[:-1], last_lora_a[1].shape[-1]
                size = prod(shape)
                if '_exps.' in name:
                    if len(shape) >= 3:
                        expert_count = shape[-2 if '.bias' in name else -3]
                        expert_params += size // expert_count
                        expert_sum += expert_count
                        n_expert_tensors += 1
                    else:
                        shared_params += size
                else:
                    shared_params += size
                total_params += size
        expert_count = (expert_sum // n_expert_tensors if n_expert_tensors >
            0 else 0)
        if last_lora_a is not None:
            total_params = -total_params
        return total_params, shared_params, expert_params, expert_count
    def format_shard_names(self, path):
        if len(self.tensors) == 1:
            return [path]
        return [path.with_name(SHARD_NAME_FORMAT.format(path.stem, i + 1,
            len(self.tensors))) for i in range(len(self.tensors))]
    def open_output_file(self, path=None):
        if self.state is WriterState.EMPTY and self.fout is not None and (
            path is None or path == self.path):
            return
        if self.state is not WriterState.NO_FILE:
            raise ValueError(
                f'Expected output file to be not yet opened, got {self.state}')
        if path is not None:
            self.path = path
        if self.path is not None:
            filenames = self.print_plan()
            self.fout = [open(filename, 'wb') for filename in filenames]
            self.state = WriterState.EMPTY
    def print_plan(self):
        logger.info('Writing the following files:')
        assert self.path is not None
        filenames = self.format_shard_names(self.path)
        assert len(filenames) == len(self.tensors)
        for name, tensors in zip(filenames, self.tensors):
            logger.info(
                f'{name}: n_tensors = {len(tensors)}, total_size = {GGUFWriter.format_n_bytes_to_str(sum(ti.nbytes for ti in tensors.values()))}'
                )
        if self.dry_run:
            logger.info('Dry run, not writing files')
            for name in filenames:
                print(name)
            exit()
        return filenames
    def add_shard_kv_data(self):
        if len(self.tensors) == 1:
            return
        total_tensors = sum(len(t) for t in self.tensors)
        assert self.fout is not None
        total_splits = len(self.fout)
        self.kv_data.extend({} for _ in range(len(self.kv_data), total_splits))
        for i, kv_data in enumerate(self.kv_data):
            kv_data[Keys.Split.LLM_KV_SPLIT_NO] = GGUFValue(i,
                GGUFValueType.UINT16)
            kv_data[Keys.Split.LLM_KV_SPLIT_COUNT] = GGUFValue(total_splits,
                GGUFValueType.UINT16)
            kv_data[Keys.Split.LLM_KV_SPLIT_TENSORS_COUNT] = GGUFValue(
                total_tensors, GGUFValueType.INT32)
    def write_header_to_file(self, path=None):
        if len(self.tensors) == 1 and (self.split_max_tensors != 0 or self.
            split_max_size != 0):
            logger.warning('Model fails split requirements, not splitting')
        self.open_output_file(path)
        if self.state is not WriterState.EMPTY:
            raise ValueError(
                f'Expected output file to be empty, got {self.state}')
        assert self.fout is not None
        assert len(self.fout) == len(self.tensors)
        assert len(self.kv_data) == 1
        self.add_shard_kv_data()
        for fout, tensors, kv_data in zip(self.fout, self.tensors, self.kv_data
            ):
            fout.write(self._pack('<I', GGUF_MAGIC, skip_pack_prefix=True))
            fout.write(self._pack('I', GGUF_VERSION))
            fout.write(self._pack('Q', len(tensors)))
            fout.write(self._pack('Q', len(kv_data)))
            fout.flush()
        self.state = WriterState.HEADER
    def write_kv_data_to_file(self):
        if self.state is not WriterState.HEADER:
            raise ValueError(
                f'Expected output file to contain the header, got {self.state}'
                )
        assert self.fout is not None
        for fout, kv_data in zip(self.fout, self.kv_data):
            kv_bytes = bytearray()
            for key, val in kv_data.items():
                kv_bytes += self._pack_val(key, GGUFValueType.STRING,
                    add_vtype=False)
                kv_bytes += self._pack_val(val.value, val.type, add_vtype=
                    True, sub_type=val.sub_type)
            fout.write(kv_bytes)
        self.flush()
        self.state = WriterState.KV_DATA
    def write_ti_data_to_file(self):
        if self.state is not WriterState.KV_DATA:
            raise ValueError(
                f'Expected output file to contain KV data, got {self.state}')
        assert self.fout is not None
        for fout, tensors in zip(self.fout, self.tensors):
            ti_data = bytearray()
            offset_tensor = 0
            for name, ti in tensors.items():
                ti_data += self._pack_val(name, GGUFValueType.STRING,
                    add_vtype=False)
                n_dims = len(ti.shape)
                ti_data += self._pack('I', n_dims)
                for j in range(n_dims):
                    ti_data += self._pack('Q', ti.shape[n_dims - 1 - j])
                ti_data += self._pack('I', ti.dtype)
                ti_data += self._pack('Q', offset_tensor)
                offset_tensor += GGUFWriter.ggml_pad(ti.nbytes, self.
                    data_alignment)
            fout.write(ti_data)
            fout.flush()
        self.state = WriterState.TI_DATA
    def add_key_value(self, key, val, vtype, sub_type=None):
        if any(key in kv_data for kv_data in self.kv_data):
            logger.warning(
                f'Duplicated key name {key!r}, overwriting it with new value {val!r} of type {vtype.name}'
                )
        self.kv_data[0][key] = GGUFValue(value=val, type=vtype, sub_type=
            sub_type)
    def add_uint8(self, key, val):
        self.add_key_value(key, val, GGUFValueType.UINT8)
    def add_int8(self, key, val):
        self.add_key_value(key, val, GGUFValueType.INT8)
    def add_uint16(self, key, val):
        self.add_key_value(key, val, GGUFValueType.UINT16)
    def add_int16(self, key, val):
        self.add_key_value(key, val, GGUFValueType.INT16)
    def add_uint32(self, key, val):
        self.add_key_value(key, val, GGUFValueType.UINT32)
    def add_int32(self, key, val):
        self.add_key_value(key, val, GGUFValueType.INT32)
    def add_float32(self, key, val):
        self.add_key_value(key, val, GGUFValueType.FLOAT32)
    def add_uint64(self, key, val):
        self.add_key_value(key, val, GGUFValueType.UINT64)
    def add_int64(self, key, val):
        self.add_key_value(key, val, GGUFValueType.INT64)
    def add_float64(self, key, val):
        self.add_key_value(key, val, GGUFValueType.FLOAT64)
    def add_bool(self, key, val):
        self.add_key_value(key, val, GGUFValueType.BOOL)
    def add_string(self, key, val):
        if not val:
            return
        self.add_key_value(key, val, GGUFValueType.STRING)
    def add_array(self, key, val):
        if len(val) == 0:
            return
        self.add_key_value(key, val, GGUFValueType.ARRAY)
    @staticmethod
    def ggml_pad(x, n):
        return (x + n - 1) // n * n
    def add_tensor_info(self, name, tensor_shape, tensor_dtype,
        tensor_nbytes, raw_dtype=None):
        if self.state is not WriterState.NO_FILE:
            raise ValueError(
                f'Expected output file to be not yet opened, got {self.state}')
        if any(name in tensors for tensors in self.tensors):
            raise ValueError(f'Duplicated tensor name {name!r}')
        if raw_dtype is None:
            if tensor_dtype == np.float16:
                dtype = GGMLQuantizationType.F16
            elif tensor_dtype == np.float32:
                dtype = GGMLQuantizationType.F32
            elif tensor_dtype == np.float64:
                dtype = GGMLQuantizationType.F64
            elif tensor_dtype == np.int8:
                dtype = GGMLQuantizationType.I8
            elif tensor_dtype == np.int16:
                dtype = GGMLQuantizationType.I16
            elif tensor_dtype == np.int32:
                dtype = GGMLQuantizationType.I32
            elif tensor_dtype == np.int64:
                dtype = GGMLQuantizationType.I64
            else:
                raise ValueError(
                    'Only F16, F32, F64, I8, I16, I32, I64 tensors are supported for now'
                    )
        else:
            dtype = raw_dtype
            if tensor_dtype == np.uint8:
                tensor_shape = quant_shape_from_byte_shape(tensor_shape,
                    raw_dtype)
        if len(self.tensors[-1]) > 0:
            if self.split_max_tensors != 0 and len(self.tensors[-1]
                ) >= self.split_max_tensors or self.split_max_size != 0 and sum(
                ti.nbytes for ti in self.tensors[-1].values()
                ) + tensor_nbytes > self.split_max_size:
                self.tensors.append({})
        self.tensors[-1][name] = TensorInfo(shape=tensor_shape, dtype=dtype,
            nbytes=tensor_nbytes)
    def add_tensor(self, name, tensor, raw_shape=None, raw_dtype=None,
        tensor_endianess=None):
        if tensor_endianess is None:
            tensor_endianess = (GGUFEndian.BIG if sys.byteorder == 'big' else
                GGUFEndian.LITTLE)
        if tensor_endianess != self.endianess:
            tensor = tensor.byteswap(inplace=False)
        if self.use_temp_file and self.temp_file is None:
            fp = tempfile.SpooledTemporaryFile(mode='w+b', max_size=256 * 
                1024 * 1024)
            fp.seek(0)
            self.temp_file = fp
        shape: Sequence[int
            ] = raw_shape if raw_shape is not None else tensor.shape
        self.add_tensor_info(name, shape, tensor.dtype, tensor.nbytes,
            raw_dtype=raw_dtype)
        if self.temp_file is None:
            self.tensors[-1][name].tensor = tensor
            return
        tensor.tofile(self.temp_file)
        self.write_padding(self.temp_file, tensor.nbytes)
    def write_padding(self, fp, n, align=None):
        pad = GGUFWriter.ggml_pad(n, align if align is not None else self.
            data_alignment) - n
        if pad != 0:
            fp.write(bytes([0] * pad))
    def write_tensor_data(self, tensor, tensor_endianess=None):
        if (self.state is not WriterState.TI_DATA and self.state is not
            WriterState.WEIGHTS):
            raise ValueError(
                f'Expected output file to contain tensor info or weights, got {self.state}'
                )
        assert self.fout is not None
        if tensor_endianess is None:
            tensor_endianess = (GGUFEndian.BIG if sys.byteorder == 'big' else
                GGUFEndian.LITTLE)
        if tensor_endianess != self.endianess:
            tensor = tensor.byteswap(inplace=False)
        file_id = -1
        for i, tensors in enumerate(self.tensors):
            if len(tensors) > 0:
                file_id = i
                break
        fout = self.fout[file_id]
        first_tensor_name = next(iter(self.tensors[file_id]))
        ti = self.tensors[file_id].pop(first_tensor_name)
        assert ti.nbytes == tensor.nbytes
        self.write_padding(fout, fout.tell())
        tensor.tofile(fout)
        self.write_padding(fout, tensor.nbytes)
        self.state = WriterState.WEIGHTS
    def write_tensors_to_file(self, *, progress: bool=False):
        self.write_ti_data_to_file()
        assert self.fout is not None
        for fout in self.fout:
            self.write_padding(fout, fout.tell())
        if self.temp_file is None:
            shard_bar = None
            bar = None
            if progress:
                from tqdm import tqdm
                total_bytes = sum(ti.nbytes for t in self.tensors for ti in
                    t.values())
                if len(self.fout) > 1:
                    shard_bar = tqdm(desc=f'Shard (0/{len(self.fout)})',
                        total=None, unit='byte', unit_scale=True)
                bar = tqdm(desc='Writing', total=total_bytes, unit='byte',
                    unit_scale=True)
            for i, (fout, tensors) in enumerate(zip(self.fout, self.tensors)):
                if shard_bar is not None:
                    shard_bar.set_description(
                        f'Shard ({i + 1}/{len(self.fout)})')
                    total = sum(ti.nbytes for ti in tensors.values())
                    shard_bar.reset(total=total if total > 0 else None)
                for ti in tensors.values():
                    assert ti.tensor is not None
                    assert ti.tensor.nbytes == ti.nbytes
                    ti.tensor.tofile(fout)
                    if shard_bar is not None:
                        shard_bar.update(ti.nbytes)
                    if bar is not None:
                        bar.update(ti.nbytes)
                    self.write_padding(fout, ti.nbytes)
                    ti.tensor = None
        else:
            self.temp_file.seek(0)
            shutil.copyfileobj(self.temp_file, self.fout[0 if not self.
                small_first_shard else 1])
            self.flush()
            self.temp_file.close()
        self.state = WriterState.WEIGHTS
    def flush(self):
        assert self.fout is not None
        for fout in self.fout:
            fout.flush()
    def close(self):
        if self.fout is not None:
            for fout in self.fout:
                fout.close()
            self.fout = None
    def add_type(self, type_name):
        self.add_string(Keys.General.TYPE, type_name)
    def add_architecture(self):
        self.add_string(Keys.General.ARCHITECTURE, self.arch)
    def add_quantization_version(self, quantization_version):
        self.add_uint32(Keys.General.QUANTIZATION_VERSION, quantization_version
            )
    def add_custom_alignment(self, alignment):
        if alignment <= 0 or alignment & alignment - 1 != 0:
            raise ValueError(
                'Invalid alignment: must be a non-zero power of two')
        self.data_alignment = alignment
        self.add_uint32(Keys.General.ALIGNMENT, alignment)
    def add_file_type(self, ftype):
        self.add_uint32(Keys.General.FILE_TYPE, ftype)
    def add_sampling_sequence(self, sequence):
        self.add_string(Keys.General.SAMPLING_SEQUENCE, sequence)
    def add_sampling_top_k(self, top_k):
        self.add_int32(Keys.General.SAMPLING_TOP_K, top_k)
    def add_sampling_top_p(self, top_p):
        self.add_float32(Keys.General.SAMPLING_TOP_P, top_p)
    def add_sampling_min_p(self, min_p):
        self.add_float32(Keys.General.SAMPLING_MIN_P, min_p)
    def add_sampling_xtc_probability(self, xtc_probability):
        self.add_float32(Keys.General.SAMPLING_XTC_PROBABILITY, xtc_probability
            )
    def add_sampling_xtc_threshold(self, xtc_threshold):
        self.add_float32(Keys.General.SAMPLING_XTC_THRESHOLD, xtc_threshold)
    def add_sampling_temp(self, temp):
        self.add_float32(Keys.General.SAMPLING_TEMP, temp)
    def add_sampling_penalty_last_n(self, penalty_last_n):
        self.add_int32(Keys.General.SAMPLING_PENALTY_LAST_N, penalty_last_n)
    def add_sampling_penalty_repeat(self, penalty_repeat):
        self.add_float32(Keys.General.SAMPLING_PENALTY_REPEAT, penalty_repeat)
    def add_sampling_mirostat(self, mirostat):
        self.add_int32(Keys.General.SAMPLING_MIROSTAT, mirostat)
    def add_sampling_mirostat_tau(self, mirostat_tau):
        self.add_float32(Keys.General.SAMPLING_MIROSTAT_TAU, mirostat_tau)
    def add_sampling_mirostat_eta(self, mirostat_eta):
        self.add_float32(Keys.General.SAMPLING_MIROSTAT_ETA, mirostat_eta)
    def add_name(self, name):
        self.add_string(Keys.General.NAME, name)
    def add_author(self, author):
        self.add_string(Keys.General.AUTHOR, author)
    def add_version(self, version):
        self.add_string(Keys.General.VERSION, version)
    def add_organization(self, organization):
        self.add_string(Keys.General.ORGANIZATION, organization)
    def add_finetune(self, finetune):
        self.add_string(Keys.General.FINETUNE, finetune)
    def add_basename(self, basename):
        self.add_string(Keys.General.BASENAME, basename)
    def add_description(self, description):
        self.add_string(Keys.General.DESCRIPTION, description)
    def add_quantized_by(self, quantized):
        self.add_string(Keys.General.QUANTIZED_BY, quantized)
    def add_size_label(self, size_label):
        self.add_string(Keys.General.SIZE_LABEL, size_label)
    def add_license(self, license):
        self.add_string(Keys.General.LICENSE, license)
    def add_license_name(self, license):
        self.add_string(Keys.General.LICENSE_NAME, license)
    def add_license_link(self, license):
        self.add_string(Keys.General.LICENSE_LINK, license)
    def add_url(self, url):
        self.add_string(Keys.General.URL, url)
    def add_doi(self, doi):
        self.add_string(Keys.General.DOI, doi)
    def add_uuid(self, uuid):
        self.add_string(Keys.General.UUID, uuid)
    def add_repo_url(self, repo_url):
        self.add_string(Keys.General.REPO_URL, repo_url)
    def add_source_url(self, url):
        self.add_string(Keys.General.SOURCE_URL, url)
    def add_source_doi(self, doi):
        self.add_string(Keys.General.SOURCE_DOI, doi)
    def add_source_uuid(self, uuid):
        self.add_string(Keys.General.SOURCE_UUID, uuid)
    def add_source_repo_url(self, repo_url):
        self.add_string(Keys.General.SOURCE_REPO_URL, repo_url)
    def add_base_model_count(self, source_count):
        self.add_uint32(Keys.General.BASE_MODEL_COUNT, source_count)
    def add_base_model_name(self, source_id, name):
        self.add_string(Keys.General.BASE_MODEL_NAME.format(id=source_id), name
            )
    def add_base_model_author(self, source_id, author):
        self.add_string(Keys.General.BASE_MODEL_AUTHOR.format(id=source_id),
            author)
    def add_base_model_version(self, source_id, version):
        self.add_string(Keys.General.BASE_MODEL_VERSION.format(id=source_id
            ), version)
    def add_base_model_organization(self, source_id, organization):
        self.add_string(Keys.General.BASE_MODEL_ORGANIZATION.format(id=
            source_id), organization)
    def add_base_model_description(self, source_id, description):
        self.add_string(Keys.General.BASE_MODEL_DESCRIPTION.format(id=
            source_id), description)
    def add_base_model_url(self, source_id, url):
        self.add_string(Keys.General.BASE_MODEL_URL.format(id=source_id), url)
    def add_base_model_doi(self, source_id, doi):
        self.add_string(Keys.General.BASE_MODEL_DOI.format(id=source_id), doi)
    def add_base_model_uuid(self, source_id, uuid):
        self.add_string(Keys.General.BASE_MODEL_UUID.format(id=source_id), uuid
            )
    def add_base_model_repo_url(self, source_id, repo_url):
        self.add_string(Keys.General.BASE_MODEL_REPO_URL.format(id=
            source_id), repo_url)
    def add_dataset_count(self, source_count):
        self.add_uint32(Keys.General.DATASET_COUNT, source_count)
    def add_dataset_name(self, source_id, name):
        self.add_string(Keys.General.DATASET_NAME.format(id=source_id), name)
    def add_dataset_author(self, source_id, author):
        self.add_string(Keys.General.DATASET_AUTHOR.format(id=source_id),
            author)
    def add_dataset_version(self, source_id, version):
        self.add_string(Keys.General.DATASET_VERSION.format(id=source_id),
            version)
    def add_dataset_organization(self, source_id, organization):
        self.add_string(Keys.General.DATASET_ORGANIZATION.format(id=
            source_id), organization)
    def add_dataset_description(self, source_id, description):
        self.add_string(Keys.General.DATASET_DESCRIPTION.format(id=
            source_id), description)
    def add_dataset_url(self, source_id, url):
        self.add_string(Keys.General.DATASET_URL.format(id=source_id), url)
    def add_dataset_doi(self, source_id, doi):
        self.add_string(Keys.General.DATASET_DOI.format(id=source_id), doi)
    def add_dataset_uuid(self, source_id, uuid):
        self.add_string(Keys.General.DATASET_UUID.format(id=source_id), uuid)
    def add_dataset_repo_url(self, source_id, repo_url):
        self.add_string(Keys.General.DATASET_REPO_URL.format(id=source_id),
            repo_url)
    def add_tags(self, tags):
        self.add_array(Keys.General.TAGS, tags)
    def add_languages(self, languages):
        self.add_array(Keys.General.LANGUAGES, languages)
    def add_tensor_data_layout(self, layout):
        self.add_string(Keys.LLM.TENSOR_DATA_LAYOUT.format(arch=self.arch),
            layout)
    def add_vocab_size(self, size):
        self.add_uint32(Keys.LLM.VOCAB_SIZE.format(arch=self.arch), size)
    def add_context_length(self, length):
        self.add_uint32(Keys.LLM.CONTEXT_LENGTH.format(arch=self.arch), length)
    def add_embedding_length(self, length):
        self.add_uint32(Keys.LLM.EMBEDDING_LENGTH.format(arch=self.arch),
            length)
    def add_embedding_length_out(self, length):
        self.add_uint32(Keys.LLM.EMBEDDING_LENGTH_OUT.format(arch=self.arch
            ), length)
    def add_features_length(self, length):
        self.add_uint32(Keys.LLM.FEATURES_LENGTH.format(arch=self.arch), length
            )
    def add_posnet_embedding_length(self, length):
        self.add_uint32(Keys.PosNet.EMBEDDING_LENGTH.format(arch=self.arch),
            length)
    def add_posnet_block_count(self, length):
        self.add_uint32(Keys.PosNet.BLOCK_COUNT.format(arch=self.arch), length)
    def add_convnext_embedding_length(self, length):
        self.add_uint32(Keys.ConvNext.EMBEDDING_LENGTH.format(arch=self.
            arch), length)
    def add_convnext_block_count(self, length):
        self.add_uint32(Keys.ConvNext.BLOCK_COUNT.format(arch=self.arch),
            length)
    def add_shortconv_l_cache(self, length):
        self.add_uint32(Keys.ShortConv.L_CACHE.format(arch=self.arch), length)
    def add_block_count(self, length):
        self.add_uint32(Keys.LLM.BLOCK_COUNT.format(arch=self.arch), length)
    def add_leading_dense_block_count(self, length):
        self.add_uint32(Keys.LLM.LEADING_DENSE_BLOCK_COUNT.format(arch=self
            .arch), length)
    def add_full_attention_interval(self, interval):
        self.add_uint32(Keys.LLM.FULL_ATTENTION_INTERVAL.format(arch=self.
            arch), interval)
    def add_feed_forward_length(self, length):
        if isinstance(length, int):
            self.add_uint32(Keys.LLM.FEED_FORWARD_LENGTH.format(arch=self.
                arch), length)
        else:
            self.add_array(Keys.LLM.FEED_FORWARD_LENGTH.format(arch=self.
                arch), length)
    def add_expert_feed_forward_length(self, length):
        self.add_uint32(Keys.LLM.EXPERT_FEED_FORWARD_LENGTH.format(arch=
            self.arch), length)
    def add_expert_shared_feed_forward_length(self, length):
        self.add_uint32(Keys.LLM.EXPERT_SHARED_FEED_FORWARD_LENGTH.format(
            arch=self.arch), length)
    def add_expert_chunk_feed_forward_length(self, length):
        self.add_uint32(Keys.LLM.EXPERT_CHUNK_FEED_FORWARD_LENGTH.format(
            arch=self.arch), length)
    def add_parallel_residual(self, use):
        self.add_bool(Keys.LLM.USE_PARALLEL_RESIDUAL.format(arch=self.arch),
            use)
    def add_decoder_start_token_id(self, id):
        self.add_uint32(Keys.LLM.DECODER_START_TOKEN_ID.format(arch=self.
            arch), id)
    def add_decoder_block_count(self, value):
        self.add_uint32(Keys.LLM.DECODER_BLOCK_COUNT.format(arch=self.arch),
            value)
    def add_embedding_length_per_layer_input(self, value):
        self.add_uint32(Keys.LLM.EMBD_LENGTH_PER_LAYER_INP.format(arch=self
            .arch), value)
    def add_altup_active_idx(self, val):
        self.add_uint32(Keys.LLM.ALTUP_ACTIVE_IDX.format(arch=self.arch), val)
    def add_altup_num_inputs(self, val):
        self.add_uint32(Keys.LLM.ALTUP_NUM_INPUTS.format(arch=self.arch), val)
    def add_activation_sparsity_scale(self, values):
        self.add_array(Keys.LLM.ACTIVATION_SPARSITY_SCALE.format(arch=self.
            arch), values)
    def add_head_count(self, count):
        if isinstance(count, int):
            self.add_uint32(Keys.Attention.HEAD_COUNT.format(arch=self.arch
                ), count)
        else:
            self.add_array(Keys.Attention.HEAD_COUNT.format(arch=self.arch),
                count)
    def add_head_count_kv(self, count):
        if isinstance(count, int):
            self.add_uint32(Keys.Attention.HEAD_COUNT_KV.format(arch=self.
                arch), count)
        else:
            self.add_array(Keys.Attention.HEAD_COUNT_KV.format(arch=self.
                arch), count)
    def add_key_length(self, length):
        self.add_uint32(Keys.Attention.KEY_LENGTH.format(arch=self.arch),
            length)
    def add_value_length(self, length):
        self.add_uint32(Keys.Attention.VALUE_LENGTH.format(arch=self.arch),
            length)
    def add_key_length_mla(self, length):
        self.add_uint32(Keys.Attention.KEY_LENGTH_MLA.format(arch=self.arch
            ), length)
    def add_value_length_mla(self, length):
        self.add_uint32(Keys.Attention.VALUE_LENGTH_MLA.format(arch=self.
            arch), length)
    def add_key_length_swa(self, length):
        self.add_uint32(Keys.Attention.KEY_LENGTH_SWA.format(arch=self.arch
            ), length)
    def add_value_length_swa(self, length):
        self.add_uint32(Keys.Attention.VALUE_LENGTH_SWA.format(arch=self.
            arch), length)
    def add_indexer_head_count(self, count):
        self.add_uint32(Keys.Attention.Indexer.HEAD_COUNT.format(arch=self.
            arch), count)
    def add_indexer_key_length(self, length):
        self.add_uint32(Keys.Attention.Indexer.KEY_LENGTH.format(arch=self.
            arch), length)
    def add_indexer_top_k(self, top_k):
        self.add_uint32(Keys.Attention.Indexer.TOP_K.format(arch=self.arch),
            top_k)
    def add_max_alibi_bias(self, bias):
        self.add_float32(Keys.Attention.MAX_ALIBI_BIAS.format(arch=self.
            arch), bias)
    def add_clamp_kqv(self, value):
        self.add_float32(Keys.Attention.CLAMP_KQV.format(arch=self.arch), value
            )
    def add_shared_kv_layers(self, value):
        self.add_uint32(Keys.Attention.SHARED_KV_LAYERS.format(arch=self.
            arch), value)
    def add_sliding_window_pattern(self, value):
        key = Keys.Attention.SLIDING_WINDOW_PATTERN.format(arch=self.arch)
        if isinstance(value, int):
            self.add_uint32(key, value)
        else:
            self.add_array(key, value)
    def add_dense_features_dims(self, dense, in_f, out_f):
        self.add_uint32(Keys.LLM.DENSE_FEAT_IN_SIZE.format(arch=self.arch,
            dense=dense), in_f)
        self.add_uint32(Keys.LLM.DENSE_FEAT_OUT_SIZE.format(arch=self.arch,
            dense=dense), out_f)
    def add_logit_scale(self, value):
        self.add_float32(Keys.LLM.LOGIT_SCALE.format(arch=self.arch), value)
    def add_attn_logit_softcapping(self, value):
        self.add_float32(Keys.LLM.ATTN_LOGIT_SOFTCAPPING.format(arch=self.
            arch), value)
    def add_router_logit_softcapping(self, value):
        self.add_float32(Keys.LLM.ROUTER_LOGIT_SOFTCAPPING.format(arch=self
            .arch), value)
    def add_final_logit_softcapping(self, value):
        self.add_float32(Keys.LLM.FINAL_LOGIT_SOFTCAPPING.format(arch=self.
            arch), value)
    def add_expert_count(self, count):
        self.add_uint32(Keys.LLM.EXPERT_COUNT.format(arch=self.arch), count)
    def add_expert_used_count(self, count):
        self.add_uint32(Keys.LLM.EXPERT_USED_COUNT.format(arch=self.arch),
            count)
    def add_expert_shared_count(self, count):
        self.add_uint32(Keys.LLM.EXPERT_SHARED_COUNT.format(arch=self.arch),
            count)
    def add_expert_group_count(self, count):
        self.add_uint32(Keys.LLM.EXPERT_GROUP_COUNT.format(arch=self.arch),
            count)
    def add_expert_group_used_count(self, count):
        self.add_uint32(Keys.LLM.EXPERT_GROUP_USED_COUNT.format(arch=self.
            arch), count)
    def add_expert_weights_scale(self, value):
        self.add_float32(Keys.LLM.EXPERT_WEIGHTS_SCALE.format(arch=self.
            arch), value)
    def add_expert_weights_norm(self, value):
        self.add_bool(Keys.LLM.EXPERT_WEIGHTS_NORM.format(arch=self.arch),
            value)
    def add_expert_gating_func(self, value):
        self.add_uint32(Keys.LLM.EXPERT_GATING_FUNC.format(arch=self.arch),
            value.value)
    def add_swiglu_clamp_exp(self, values):
        self.add_array(Keys.LLM.SWIGLU_CLAMP_EXP.format(arch=self.arch), values
            )
    def add_swiglu_clamp_shexp(self, values):
        self.add_array(Keys.LLM.SWIGLU_CLAMP_SHEXP.format(arch=self.arch),
            values)
    def add_hidden_act(self, value):
        self.add_string(Keys.LLM.HIDDEN_ACT.format(arch=self.arch), value)
    def add_expert_group_scale(self, value):
        self.add_float32(Keys.LLM.EXPERT_GROUP_SCALE.format(arch=self.arch),
            value)
    def add_experts_per_group(self, count):
        self.add_uint32(Keys.LLM.EXPERTS_PER_GROUP.format(arch=self.arch),
            count)
    def add_moe_every_n_layers(self, value):
        self.add_uint32(Keys.LLM.MOE_EVERY_N_LAYERS.format(arch=self.arch),
            value)
    def add_moe_latent_size(self, value):
        self.add_uint32(Keys.LLM.MOE_LATENT_SIZE.format(arch=self.arch), value)
    def add_nextn_predict_layers(self, count):
        self.add_uint32(Keys.LLM.NEXTN_PREDICT_LAYERS.format(arch=self.arch
            ), count)
    def add_swin_norm(self, value):
        self.add_bool(Keys.LLM.SWIN_NORM.format(arch=self.arch), value)
    def add_rescale_every_n_layers(self, count):
        self.add_uint32(Keys.LLM.RESCALE_EVERY_N_LAYERS.format(arch=self.
            arch), count)
    def add_time_mix_extra_dim(self, dim):
        self.add_uint32(Keys.LLM.TIME_MIX_EXTRA_DIM.format(arch=self.arch), dim
            )
    def add_time_decay_extra_dim(self, dim):
        self.add_uint32(Keys.LLM.TIME_DECAY_EXTRA_DIM.format(arch=self.arch
            ), dim)
    def add_residual_scale(self, value):
        self.add_float32(Keys.LLM.RESIDUAL_SCALE.format(arch=self.arch), value)
    def add_embedding_scale(self, value):
        self.add_float32(Keys.LLM.EMBEDDING_SCALE.format(arch=self.arch), value
            )
    def add_wkv_head_size(self, size):
        self.add_uint32(Keys.WKV.HEAD_SIZE.format(arch=self.arch), size)
    def add_token_shift_count(self, count):
        self.add_uint32(Keys.LLM.TOKEN_SHIFT_COUNT.format(arch=self.arch),
            count)
    def add_interleave_moe_layer_step(self, value):
        self.add_uint32(Keys.LLM.INTERLEAVE_MOE_LAYER_STEP.format(arch=self
            .arch), value)
    def add_layer_norm_eps(self, value):
        self.add_float32(Keys.Attention.LAYERNORM_EPS.format(arch=self.arch
            ), value)
    def add_layer_norm_rms_eps(self, value):
        self.add_float32(Keys.Attention.LAYERNORM_RMS_EPS.format(arch=self.
            arch), value)
    def add_group_norm_eps(self, value):
        self.add_float32(Keys.Attention.GROUPNORM_EPS.format(arch=self.arch
            ), value)
    def add_group_norm_groups(self, value):
        self.add_uint32(Keys.Attention.GROUPNORM_GROUPS.format(arch=self.
            arch), value)
    def add_causal_attention(self, value):
        self.add_bool(Keys.Attention.CAUSAL.format(arch=self.arch), value)
    def add_q_lora_rank(self, length):
        self.add_uint32(Keys.Attention.Q_LORA_RANK.format(arch=self.arch),
            length)
    def add_kv_lora_rank(self, length):
        self.add_uint32(Keys.Attention.KV_LORA_RANK.format(arch=self.arch),
            length)
    def add_decay_lora_rank(self, length):
        self.add_uint32(Keys.Attention.DECAY_LORA_RANK.format(arch=self.
            arch), length)
    def add_iclr_lora_rank(self, length):
        self.add_uint32(Keys.Attention.ICLR_LORA_RANK.format(arch=self.arch
            ), length)
    def add_value_residual_mix_lora_rank(self, length):
        self.add_uint32(Keys.Attention.VALUE_RESIDUAL_MIX_LORA_RANK.format(
            arch=self.arch), length)
    def add_rope_freq_base_swa(self, value):
        self.add_float32(Keys.Rope.FREQ_BASE_SWA.format(arch=self.arch), value)
    def add_gate_lora_rank(self, length):
        self.add_uint32(Keys.Attention.GATE_LORA_RANK.format(arch=self.arch
            ), length)
    def add_relative_attn_buckets_count(self, value):
        self.add_uint32(Keys.Attention.REL_BUCKETS_COUNT.format(arch=self.
            arch), value)
    def add_sliding_window(self, value):
        self.add_uint32(Keys.Attention.SLIDING_WINDOW.format(arch=self.arch
            ), value)
    def add_attention_scale(self, value):
        self.add_float32(Keys.Attention.SCALE.format(arch=self.arch), value)
    def add_attn_output_scale(self, value):
        self.add_float32(Keys.Attention.OUTPUT_SCALE.format(arch=self.arch),
            value)
    def add_attn_value_scale(self, value):
        self.add_float32(Keys.Attention.VALUE_SCALE.format(arch=self.arch),
            value)
    def add_attn_temperature_length(self, value):
        self.add_uint32(Keys.Attention.TEMPERATURE_LENGTH.format(arch=self.
            arch), value)
    def add_attn_temperature_scale(self, value):
        self.add_float32(Keys.Attention.TEMPERATURE_SCALE.format(arch=self.
            arch), value)
    def add_pooling_type(self, value):
        self.add_uint32(Keys.LLM.POOLING_TYPE.format(arch=self.arch), value
            .value)
    def add_num_deepstack_layers(self, count):
        """Add scalar deepstack layer count (qwen3vl format)"""
        self.add_uint32(Keys.LLM.NUM_DEEPSTACK_LAYERS.format(arch=self.arch
            ), count)
    def add_deepstack_mapping(self, layers):
        """Add per-layer deepstack projector indices (Granite4 Vision format)"""
        self.add_array(Keys.LLM.DEEPSTACK_MAPPING.format(arch=self.arch),
            list(layers))
    def add_rope_dimension_count(self, count):
        self.add_uint32(Keys.Rope.DIMENSION_COUNT.format(arch=self.arch), count
            )
    def add_rope_dimension_count_swa(self, count):
        self.add_uint32(Keys.Rope.DIMENSION_COUNT_SWA.format(arch=self.arch
            ), count)
    def add_rope_dimension_sections(self, dims):
        self.add_array(Keys.Rope.DIMENSION_SECTIONS.format(arch=self.arch),
            dims)
    def add_rope_freq_base(self, value):
        self.add_float32(Keys.Rope.FREQ_BASE.format(arch=self.arch), value)
    def add_rope_scaling_type(self, value):
        self.add_string(Keys.Rope.SCALING_TYPE.format(arch=self.arch),
            value.value)
    def add_rope_scaling_factor(self, value):
        self.add_float32(Keys.Rope.SCALING_FACTOR.format(arch=self.arch), value
            )
    def add_rope_scaling_alpha(self, value):
        self.add_float32(Keys.Rope.SCALING_ALPHA.format(arch=self.arch), value)
    def add_rope_scaling_attn_factors(self, value):
        self.add_float32(Keys.Rope.SCALING_ATTN_FACTOR.format(arch=self.
            arch), value)
    def add_rope_scaling_orig_ctx_len(self, value):
        self.add_uint32(Keys.Rope.SCALING_ORIG_CTX_LEN.format(arch=self.
            arch), value)
    def add_rope_scaling_finetuned(self, value):
        self.add_bool(Keys.Rope.SCALING_FINETUNED.format(arch=self.arch), value
            )
    def add_rope_scaling_yarn_log_mul(self, value):
        self.add_float32(Keys.Rope.SCALING_YARN_LOG_MUL.format(arch=self.
            arch), value)
    def add_rope_scaling_yarn_ext_factor(self, value):
        self.add_float32(Keys.Rope.SCALING_YARN_EXT_FACTOR.format(arch=self
            .arch), value)
    def add_rope_scaling_yarn_attn_factor(self, value):
        self.add_float32(Keys.Rope.SCALING_YARN_ATTN_FACTOR.format(arch=
            self.arch), value)
    def add_rope_scaling_yarn_beta_fast(self, value):
        self.add_float32(Keys.Rope.SCALING_YARN_BETA_FAST.format(arch=self.
            arch), value)
    def add_rope_scaling_yarn_beta_slow(self, value):
        self.add_float32(Keys.Rope.SCALING_YARN_BETA_SLOW.format(arch=self.
            arch), value)
    def add_ssm_conv_kernel(self, value):
        self.add_uint32(Keys.SSM.CONV_KERNEL.format(arch=self.arch), value)
    def add_ssm_inner_size(self, value):
        self.add_uint32(Keys.SSM.INNER_SIZE.format(arch=self.arch), value)
    def add_ssm_state_size(self, value):
        self.add_uint32(Keys.SSM.STATE_SIZE.format(arch=self.arch), value)
    def add_ssm_time_step_rank(self, value):
        self.add_uint32(Keys.SSM.TIME_STEP_RANK.format(arch=self.arch), value)
    def add_ssm_group_count(self, value):
        self.add_uint32(Keys.SSM.GROUP_COUNT.format(arch=self.arch), value)
    def add_ssm_dt_b_c_rms(self, value):
        self.add_bool(Keys.SSM.DT_B_C_RMS.format(arch=self.arch), value)
    def add_kda_head_dim(self, value):
        self.add_uint32(Keys.KDA.HEAD_DIM.format(arch=self.arch), value)
    def add_tokenizer_model(self, model):
        self.add_string(Keys.Tokenizer.MODEL, model)
    def add_tokenizer_pre(self, pre):
        self.add_string(Keys.Tokenizer.PRE, pre)
    def add_token_list(self, tokens):
        self.add_array(Keys.Tokenizer.LIST, tokens)
    def add_token_merges(self, merges):
        self.add_array(Keys.Tokenizer.MERGES, merges)
    def add_token_types(self, types):
        self.add_array(Keys.Tokenizer.TOKEN_TYPE, types)
    def add_token_type_count(self, value):
        self.add_uint32(Keys.Tokenizer.TOKEN_TYPE_COUNT, value)
    def add_token_scores(self, scores):
        self.add_array(Keys.Tokenizer.SCORES, scores)
    def add_bos_token_id(self, id):
        self.add_uint32(Keys.Tokenizer.BOS_ID, id)
    def add_eos_token_id(self, id):
        self.add_uint32(Keys.Tokenizer.EOS_ID, id)
    def add_unk_token_id(self, id):
        self.add_uint32(Keys.Tokenizer.UNK_ID, id)
    def add_sep_token_id(self, id):
        self.add_uint32(Keys.Tokenizer.SEP_ID, id)
    def add_pad_token_id(self, id):
        self.add_uint32(Keys.Tokenizer.PAD_ID, id)
    def add_mask_token_id(self, id):
        self.add_uint32(Keys.Tokenizer.MASK_ID, id)
    def add_add_bos_token(self, value):
        self.add_bool(Keys.Tokenizer.ADD_BOS, value)
    def add_add_eos_token(self, value):
        self.add_bool(Keys.Tokenizer.ADD_EOS, value)
    def add_add_sep_token(self, value):
        self.add_bool(Keys.Tokenizer.ADD_SEP, value)
    def add_add_space_prefix(self, value):
        self.add_bool(Keys.Tokenizer.ADD_PREFIX, value)
    def add_remove_extra_whitespaces(self, value):
        self.add_bool(Keys.Tokenizer.REMOVE_EXTRA_WS, value)
    def add_precompiled_charsmap(self, charsmap):
        self.add_array(Keys.Tokenizer.PRECOMPILED_CHARSMAP, charsmap)
    def add_chat_template(self, value):
        if not isinstance(value, str):
            template_default = None
            template_names = set()
            for choice in value:
                name = choice.get('name', '')
                template = choice.get('template')
                name = ''.join(c if c in ascii_letters + digits else '_' for
                    c in name)
                if name and template is not None:
                    if name == 'default':
                        template_default = template
                    else:
                        template_names.add(name)
                        self.add_string(Keys.Tokenizer.CHAT_TEMPLATE_N.
                            format(name=name), template)
            if template_names:
                self.add_array(Keys.Tokenizer.CHAT_TEMPLATES, list(
                    template_names))
            if template_default is None:
                return
            value = template_default
        self.add_string(Keys.Tokenizer.CHAT_TEMPLATE, value)
    def add_suppress_tokens(self, tokens):
        self.add_array(Keys.Tokenizer.SUPPRESS_TOKENS, tokens)
    def add_normalizer_lowercase(self, value):
        self.add_bool(Keys.Tokenizer.NORMALIZER_LOWERCASE, value)
    def add_normalizer_strip_accents(self, value):
        self.add_bool(Keys.Tokenizer.NORMALIZER_STRIP_ACCENTS, value)
    def add_eot_token_id(self, id):
        self.add_uint32(Keys.Tokenizer.EOT_ID, id)
    def add_eom_token_id(self, id):
        self.add_uint32(Keys.Tokenizer.EOM_ID, id)
    def add_classifier_output_labels(self, labels):
        self.add_array(Keys.Classifier.OUTPUT_LABELS.format(arch=self.arch),
            labels)
    def add_clip_has_vision_encoder(self, value):
        self.add_bool(Keys.Clip.HAS_VISION_ENCODER, value)
    def add_clip_has_audio_encoder(self, value):
        self.add_bool(Keys.Clip.HAS_AUDIO_ENCODER, value)
    def add_clip_projector_type(self, value):
        self.add_string(Keys.Clip.PROJECTOR_TYPE, value)
    def add_clip_vision_projector_type(self, value):
        self.add_string(Keys.ClipVision.PROJECTOR_TYPE, value)
    def add_vision_projection_dim(self, value):
        self.add_uint32(Keys.ClipVision.PROJECTION_DIM, value)
    def add_vision_patch_size(self, value):
        self.add_uint32(Keys.ClipVision.PATCH_SIZE, value)
    def add_vision_embedding_length(self, value):
        self.add_uint32(Keys.ClipVision.EMBEDDING_LENGTH, value)
    def add_vision_feed_forward_length(self, value):
        self.add_uint32(Keys.ClipVision.FEED_FORWARD_LENGTH, value)
    def add_vision_block_count(self, value):
        self.add_uint32(Keys.ClipVision.BLOCK_COUNT, value)
    def add_vision_head_count(self, value):
        self.add_uint32(Keys.ClipVision.Attention.HEAD_COUNT, value)
    def add_vision_head_count_kv(self, value):
        self.add_uint32(Keys.ClipVision.Attention.HEAD_COUNT_KV, value)
    def add_vision_attention_layernorm_eps(self, value):
        self.add_float32(Keys.ClipVision.Attention.LAYERNORM_EPS, value)
    def add_vision_image_size(self, value):
        self.add_uint32(Keys.ClipVision.IMAGE_SIZE, value)
    def add_vision_max_pixels(self, value):
        self.add_uint32(Keys.ClipVision.IMAGE_MAX_PIXELS, value)
    def add_vision_min_pixels(self, value):
        self.add_uint32(Keys.ClipVision.IMAGE_MIN_PIXELS, value)
    def add_vision_preproc_max_tiles(self, value):
        self.add_uint32(Keys.ClipVision.PREPROC_MAX_TILES, value)
    def add_vision_preproc_min_tiles(self, value):
        self.add_uint32(Keys.ClipVision.PREPROC_MIN_TILES, value)
    def add_vision_preproc_image_size(self, value):
        self.add_uint32(Keys.ClipVision.PREPROC_IMAGE_SIZE, value)
    def add_vision_projector_query_side(self, value):
        self.add_uint32(Keys.ClipVision.Projector.QUERY_SIDE, value)
    def add_vision_projector_window_side(self, value):
        self.add_uint32(Keys.ClipVision.Projector.WINDOW_SIDE, value)
    def add_vision_spatial_offsets(self, layers):
        self.add_array(Keys.ClipVision.Projector.SPATIAL_OFFSETS, layers)
    def add_vision_image_mean(self, values):
        self.add_array(Keys.ClipVision.IMAGE_MEAN, values)
    def add_vision_image_std(self, values):
        self.add_array(Keys.ClipVision.IMAGE_STD, values)
    def add_vision_spatial_merge_size(self, value):
        self.add_uint32(Keys.ClipVision.SPATIAL_MERGE_SIZE, value)
    def add_vision_use_gelu(self, value):
        self.add_bool(Keys.ClipVision.USE_GELU, value)
    def add_vision_use_silu(self, value):
        self.add_bool(Keys.ClipVision.USE_SILU, value)
    def add_vision_projector_scale_factor(self, value):
        self.add_uint32(Keys.ClipVision.Projector.SCALE_FACTOR, value)
    def add_vision_n_wa_pattern(self, value):
        """Add window attention pattern interval for vision models.
        This defines the pattern interval for window attention vs full attention layers.
        For example, if n_wa_pattern=4, then layers 3, 7, 11, ... use full attention,
        while other layers use window attention.
        Used by models like Qwen2.5-VL where full attention layers follow a regular pattern.
        """
        self.add_uint32(Keys.ClipVision.N_WA_PATTERN, value)
    def add_vision_wa_layer_indexes(self, layers):
        """Add explicit layer indexes that use full attention in vision models.
        This specifies the exact layer indices (0-based) that should use full attention
        instead of window attention. All other layers will use window attention.
        Args:
            layers: List of layer indices that use full attention (e.g., [3, 7, 11, 15])
        Used by models like YoutuVL where full attention layers are explicitly specified
        rather than following a regular pattern.
        Difference from add_vision_n_wa_pattern:
        - n_wa_pattern: Defines a regular interval pattern (every Nth layer uses full attention)
        - wa_layer_indexes: Explicitly lists which layers use full attention (irregular pattern)
        """
        self.add_array(Keys.ClipVision.WA_LAYER_INDEXES, layers)
    def add_vision_is_deepstack_layers(self, layers):
        self.add_array(Keys.ClipVision.IS_DEEPSTACK_LAYERS, layers)
    def add_vision_wa_pattern_mode(self, modes):
        self.add_array(Keys.ClipVision.WA_PATTERN_MODE, modes)
    def add_vision_window_size(self, value):
        self.add_uint32(Keys.ClipVision.WINDOW_SIZE, value)
    def add_vision_feature_layers(self, layers):
        self.add_array(Keys.ClipVision.FEATURE_LAYERS, layers)
    def add_vision_image_grid_pinpoints(self, layers):
        self.add_array(Keys.ClipVision.IMAGE_GRID_PINPOINTS, layers)
    def add_vision_sam_layers_count(self, value):
        self.add_uint32(Keys.ClipVision.SAM.BLOCK_COUNT, value)
    def add_vision_sam_embedding_length(self, value):
        self.add_uint32(Keys.ClipVision.SAM.EMBEDDING_LENGTH, value)
    def add_vision_sam_head_count(self, value):
        self.add_uint32(Keys.ClipVision.SAM.HEAD_COUNT, value)
    def add_clip_audio_projector_type(self, value):
        self.add_string(Keys.ClipAudio.PROJECTOR_TYPE, value)
    def add_audio_projection_dim(self, value):
        self.add_uint32(Keys.ClipAudio.PROJECTION_DIM, value)
    def add_audio_embedding_length(self, value):
        self.add_uint32(Keys.ClipAudio.EMBEDDING_LENGTH, value)
    def add_audio_feed_forward_length(self, value):
        self.add_uint32(Keys.ClipAudio.FEED_FORWARD_LENGTH, value)
    def add_audio_block_count(self, value):
        self.add_uint32(Keys.ClipAudio.BLOCK_COUNT, value)
    def add_audio_head_count(self, value):
        self.add_uint32(Keys.ClipAudio.Attention.HEAD_COUNT, value)
    def add_audio_attention_layernorm_eps(self, value):
        self.add_float32(Keys.ClipAudio.Attention.LAYERNORM_EPS, value)
    def add_audio_num_mel_bins(self, value):
        self.add_uint32(Keys.ClipAudio.NUM_MEL_BINS, value)
    def add_audio_stack_factor(self, value):
        self.add_uint32(Keys.ClipAudio.Projector.STACK_FACTOR, value)
    def add_audio_chunk_size(self, value):
        self.add_uint32(Keys.ClipAudio.CHUNK_SIZE, value)
    def add_audio_conv_kernel_size(self, value):
        self.add_uint32(Keys.ClipAudio.CONV_KERNEL_SIZE, value)
    def add_audio_max_pos_emb(self, value):
        self.add_uint32(Keys.ClipAudio.MAX_POS_EMB, value)
    def add_audio_projector_window_size(self, value):
        self.add_uint32(Keys.ClipAudio.Projector.WINDOW_SIZE, value)
    def add_audio_projector_downsample_rate(self, value):
        self.add_uint32(Keys.ClipAudio.Projector.DOWNSAMPLE_RATE, value)
    def add_audio_projector_head_count(self, value):
        self.add_uint32(Keys.ClipAudio.Projector.HEAD_COUNT, value)
    def add_xielu_alpha_p(self, values):
        self.add_array(Keys.xIELU.ALPHA_P, values)
    def add_xielu_alpha_n(self, values):
        self.add_array(Keys.xIELU.ALPHA_N, values)
    def add_xielu_beta(self, values):
        self.add_array(Keys.xIELU.BETA, values)
    def add_xielu_eps(self, values):
        self.add_array(Keys.xIELU.EPS, values)
    def add_diffusion_shift_logits(self, value):
        self.add_bool(Keys.Diffusion.SHIFT_LOGITS, value)
    def _pack(self, fmt, value, skip_pack_prefix=False):
        pack_prefix = ''
        if not skip_pack_prefix:
            pack_prefix = '<' if self.endianess == GGUFEndian.LITTLE else '>'
        return struct.pack(f'{pack_prefix}{fmt}', value)
    def _pack_val(self, val, vtype, add_vtype, sub_type=None):
        kv_data = bytearray()
        if add_vtype:
            kv_data += self._pack('I', vtype)
        pack_fmt = self._simple_value_packing.get(vtype)
        if pack_fmt is not None:
            kv_data += self._pack(pack_fmt, val, skip_pack_prefix=vtype ==
                GGUFValueType.BOOL)
        elif vtype == GGUFValueType.STRING:
            encoded_val = val.encode('utf-8') if isinstance(val, str) else val
            kv_data += self._pack('Q', len(encoded_val))
            kv_data += encoded_val
        elif vtype == GGUFValueType.ARRAY:
            if not isinstance(val, Sequence):
                raise ValueError(
                    'Invalid GGUF metadata array, expecting sequence')
            if len(val) == 0:
                raise ValueError('Invalid GGUF metadata array. Empty array')
            if sub_type is not None:
                ltype = sub_type
            elif isinstance(val, bytes):
                ltype = GGUFValueType.UINT8
            else:
                ltype = GGUFValueType.get_type(val[0])
                if not all(GGUFValueType.get_type(i) is ltype for i in val[1:]
                    ):
                    raise ValueError(
                        'All items in a GGUF array should be of the same type')
            kv_data += self._pack('I', ltype)
            kv_data += self._pack('Q', len(val))
            for item in val:
                kv_data += self._pack_val(item, ltype, add_vtype=False)
        else:
            raise ValueError('Invalid GGUF metadata value type or value')
        return bytes(kv_data)
    @staticmethod
    def format_n_bytes_to_str(num):
        if num == 0:
            return 'negligible - metadata only'
        fnum = float(num)
        for unit in ('', 'K', 'M', 'G'):
            if abs(fnum) < 1000.0:
                return f'{fnum:3.1f}{unit}'
            fnum /= 1000.0
        return f'{fnum:.1f}T - over 1TB, split recommended'