from __future__ import annotations
import logging, os, sys
from collections import OrderedDict
from typing import Any, Literal, NamedTuple, TypeVar, Union
import numpy as np
import numpy.typing as npt
from .quant5 import quant_shape_to_byte_shape
if __name__ == '__main__':
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
from .const import GGML_QUANT_SIZES, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION, GGMLQuantizationType, GGUFValueType, GGUFEndian
logger = logging.getLogger(__name__)
READER_SUPPORTED_VERSIONS = [2, GGUF_VERSION]
class ReaderField(NamedTuple):
    offset: int
    name: str
    parts: list[npt.NDArray[Any]] = []
    data: list[int] = [-1]
    types: list[GGUFValueType] = []
    def contents(self, index_or_slice=slice(None)):
        if self.types:
            to_string = lambda x: str(x.tobytes(), encoding='utf-8')
            main_type = self.types[0]
            if main_type == GGUFValueType.ARRAY:
                sub_type = self.types[-1]
                if sub_type == GGUFValueType.STRING:
                    indices = self.data[index_or_slice]
                    if isinstance(index_or_slice, int):
                        return to_string(self.parts[indices])
                    else:
                        return [to_string(self.parts[idx]) for idx in indices]
                elif isinstance(index_or_slice, int):
                    return self.parts[self.data[index_or_slice]].tolist()[0]
                else:
                    return [pv for idx in self.data[index_or_slice] for pv in
                        self.parts[idx].tolist()]
            if main_type == GGUFValueType.STRING:
                return to_string(self.parts[-1])
            else:
                return self.parts[-1].tolist()[0]
        return None
class ReaderTensor(NamedTuple):
    name: str
    tensor_type: GGMLQuantizationType
    shape: npt.NDArray[np.uint32]
    n_elements: int
    n_bytes: int
    data_offset: int
    data: npt.NDArray[Any]
    field: ReaderField
class GGUFReader:
    byte_order: Literal['I', 'S'] = 'I'
    alignment: int = GGUF_DEFAULT_ALIGNMENT
    data_offset: int
    gguf_scalar_to_np: dict[GGUFValueType, type[np.generic]] = {GGUFValueType
        .UINT8: np.uint8, GGUFValueType.INT8: np.int8, GGUFValueType.UINT16:
        np.uint16, GGUFValueType.INT16: np.int16, GGUFValueType.UINT32: np.
        uint32, GGUFValueType.INT32: np.int32, GGUFValueType.FLOAT32: np.
        float32, GGUFValueType.UINT64: np.uint64, GGUFValueType.INT64: np.
        int64, GGUFValueType.FLOAT64: np.float64, GGUFValueType.BOOL: np.bool_}
    def __init__(self, path, mode='r'):
        self.data = np.memmap(path, mode=mode)
        offs = 0
        if self._get(offs, np.uint32, override_order='<')[0] != GGUF_MAGIC:
            raise ValueError('GGUF magic invalid')
        offs += 4
        temp_version = self._get(offs, np.uint32)
        if temp_version[0] & 65535 == 0:
            self.byte_order = 'S'
            temp_version = temp_version.newbyteorder(self.byte_order)
        version = temp_version[0]
        if version not in READER_SUPPORTED_VERSIONS:
            raise ValueError(
                f'Sorry, file appears to be version {version} which we cannot handle'
                )
        if sys.byteorder == 'little':
            host_endian = GGUFEndian.LITTLE
            swapped_endian = GGUFEndian.BIG
        else:
            host_endian = GGUFEndian.BIG
            swapped_endian = GGUFEndian.LITTLE
        self.endianess = (swapped_endian if self.byte_order == 'S' else
            host_endian)
        self.fields: OrderedDict[str, ReaderField] = OrderedDict()
        self.tensors: list[ReaderTensor] = []
        offs += self._push_field(ReaderField(offs, 'GGUF.version', [
            temp_version], [0], [GGUFValueType.UINT32]))
        temp_counts = self._get(offs, np.uint64, 2)
        offs += self._push_field(ReaderField(offs, 'GGUF.tensor_count', [
            temp_counts[:1]], [0], [GGUFValueType.UINT64]))
        offs += self._push_field(ReaderField(offs, 'GGUF.kv_count', [
            temp_counts[1:]], [0], [GGUFValueType.UINT64]))
        tensor_count, kv_count = temp_counts
        offs = self._build_fields(offs, kv_count)
        offs, tensors_fields = self._build_tensor_info(offs, tensor_count)
        new_align = self.fields.get('general.alignment')
        if new_align is not None:
            if new_align.types != [GGUFValueType.UINT32]:
                raise ValueError('Bad type for general.alignment field')
            self.alignment = new_align.parts[-1][0]
        padding = offs % self.alignment
        if padding != 0:
            offs += self.alignment - padding
        self.data_offset = offs
        self._build_tensors(offs, tensors_fields)
    _DT = TypeVar('_DT', bound=npt.DTypeLike)
    def get_field(self, key):
        return self.fields.get(key, None)
    def get_tensor(self, idx):
        return self.tensors[idx]
    def _get(self, offset, dtype, count=1, override_order=None):
        count = int(count)
        itemsize = int(np.empty([], dtype=dtype).itemsize)
        end_offs = offset + itemsize * count
        return self.data[offset:end_offs].view(dtype=dtype)[:count]
    def _push_field(self, field, skip_sum=False):
        if field.name in self.fields:
            logger.warning(
                f'Duplicate key {field.name} at offset {field.offset}')
            self.fields[field.name + '_{}'.format(field.offset)] = field
        else:
            self.fields[field.name] = field
        return 0 if skip_sum else sum(int(part.nbytes) for part in field.parts)
    def _get_str(self, offset):
        slen = self._get(offset, np.uint64)
        return slen, self._get(offset + 8, np.uint8, slen[0])
    def _get_field_parts(self, orig_offs, raw_type):
        offs = orig_offs
        types: list[GGUFValueType] = []
        gtype = GGUFValueType(raw_type)
        types.append(gtype)
        if gtype == GGUFValueType.STRING:
            sparts: list[npt.NDArray[Any]] = list(self._get_str(offs))
            size = sum(int(part.nbytes) for part in sparts)
            return size, sparts, [1], types
        nptype = self.gguf_scalar_to_np.get(gtype)
        if nptype is not None:
            val = self._get(offs, nptype)
            return int(val.nbytes), [val], [0], types
        if gtype == GGUFValueType.ARRAY:
            raw_itype = self._get(offs, np.uint32)
            offs += int(raw_itype.nbytes)
            alen = self._get(offs, np.uint64)
            offs += int(alen.nbytes)
            aparts: list[npt.NDArray[Any]] = [raw_itype, alen]
            data_idxs: list[int] = []
            for idx in range(alen[0]):
                curr_size, curr_parts, curr_idxs, curr_types = (self.
                    _get_field_parts(offs, raw_itype[0]))
                if idx == 0:
                    types += curr_types
                idxs_offs = len(aparts)
                aparts += curr_parts
                data_idxs += (idx + idxs_offs for idx in curr_idxs)
                offs += curr_size
            return offs - orig_offs, aparts, data_idxs, types
        raise ValueError(f'Unknown/unhandled field type {gtype}')
    def _get_tensor_info_field(self, orig_offs):
        offs = orig_offs
        name_len, name_data = self._get_str(offs)
        offs += int(name_len.nbytes + name_data.nbytes)
        n_dims = self._get(offs, np.uint32)
        offs += int(n_dims.nbytes)
        dims = self._get(offs, np.uint64, n_dims[0])
        offs += int(dims.nbytes)
        raw_dtype = self._get(offs, np.uint32)
        offs += int(raw_dtype.nbytes)
        offset_tensor = self._get(offs, np.uint64)
        offs += int(offset_tensor.nbytes)
        return ReaderField(orig_offs, str(bytes(name_data), encoding=
            'utf-8'), [name_len, name_data, n_dims, dims, raw_dtype,
            offset_tensor], [1, 3, 4, 5])
    def _build_fields(self, offs, count):
        for _ in range(count):
            orig_offs = offs
            kv_klen, kv_kdata = self._get_str(offs)
            offs += int(kv_klen.nbytes + kv_kdata.nbytes)
            raw_kv_type = self._get(offs, np.uint32)
            offs += int(raw_kv_type.nbytes)
            parts: list[npt.NDArray[Any]] = [kv_klen, kv_kdata, raw_kv_type]
            idxs_offs = len(parts)
            field_size, field_parts, field_idxs, field_types = (self.
                _get_field_parts(offs, raw_kv_type[0]))
            parts += field_parts
            self._push_field(ReaderField(orig_offs, str(bytes(kv_kdata),
                encoding='utf-8'), parts, [(idx + idxs_offs) for idx in
                field_idxs], field_types), skip_sum=True)
            offs += field_size
        return offs
    def _build_tensor_info(self, offs, count):
        tensor_fields = []
        for _ in range(count):
            field = self._get_tensor_info_field(offs)
            offs += sum(int(part.nbytes) for part in field.parts)
            tensor_fields.append(field)
        return offs, tensor_fields
    def _build_tensors(self, start_offs, fields):
        tensors = []
        tensor_names = set()
        for field in fields:
            (_name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor
                ) = field.parts
            tensor_name = str(bytes(name_data), encoding='utf-8')
            if tensor_name in tensor_names:
                raise ValueError(
                    f'Found duplicated tensor with name {tensor_name}')
            tensor_names.add(tensor_name)
            ggml_type = GGMLQuantizationType(raw_dtype[0])
            n_elems = int(np.prod(dims))
            np_dims = tuple(reversed(dims.tolist()))
            block_size, type_size = GGML_QUANT_SIZES[ggml_type]
            n_bytes = n_elems * type_size // block_size
            data_offs = int(start_offs + offset_tensor[0])
            item_type: npt.DTypeLike
            if ggml_type == GGMLQuantizationType.F16:
                item_count = n_elems
                item_type = np.float16
            elif ggml_type == GGMLQuantizationType.F32:
                item_count = n_elems
                item_type = np.float32
            elif ggml_type == GGMLQuantizationType.F64:
                item_count = n_elems
                item_type = np.float64
            elif ggml_type == GGMLQuantizationType.I8:
                item_count = n_elems
                item_type = np.int8
            elif ggml_type == GGMLQuantizationType.I16:
                item_count = n_elems
                item_type = np.int16
            elif ggml_type == GGMLQuantizationType.I32:
                item_count = n_elems
                item_type = np.int32
            elif ggml_type == GGMLQuantizationType.I64:
                item_count = n_elems
                item_type = np.int64
            else:
                item_count = n_bytes
                item_type = np.uint8
                np_dims = quant_shape_to_byte_shape(np_dims, ggml_type)
            tensors.append(ReaderTensor(name=tensor_name, tensor_type=
                ggml_type, shape=dims, n_elements=n_elems, n_bytes=n_bytes,
                data_offset=data_offs, data=self._get(data_offs, item_type,
                item_count).reshape(np_dims), field=field))
        self.tensors = tensors