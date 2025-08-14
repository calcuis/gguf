from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence
from math import log2, ceil
from numpy.typing import DTypeLike
import numpy as np
from .const import GGML_QUANT_SIZES, GGMLQuantizationType, QK_K
from .lazy import LazyNumpyTensor
def quant_shape_to_byte_shape(shape, quant_type):
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % block_size != 0:
        raise ValueError(
            f'Quantized tensor row size ({shape[-1]}) is not a multiple of {quant_type.name} block size ({block_size})'
            )
    return *shape[:-1], shape[-1] // block_size * type_size
def quant_shape_from_byte_shape(shape, quant_type):
    block_size, type_size = GGML_QUANT_SIZES[quant_type]
    if shape[-1] % type_size != 0:
        raise ValueError(
            f'Quantized tensor bytes per row ({shape[-1]}) is not a multiple of {quant_type.name} type size ({type_size})'
            )
    return *shape[:-1], shape[-1] // type_size * block_size
def _apply_over_grouped_rows(func, arr, otype, oshape):
    rows = arr.reshape((-1, arr.shape[-1]))
    osize = 1
    for dim in oshape:
        osize *= dim
    out = np.empty(shape=osize, dtype=otype)
    n_groups = rows.shape[0] // 16 or 1
    np.concatenate([func(group).ravel() for group in np.array_split(rows,
        n_groups)], axis=0, out=out)
    return out.reshape(oshape)
def np_roundf(n):
    a = abs(n)
    floored = np.floor(a)
    b = floored + np.floor(2 * (a - floored))
    return np.sign(n) * b
class QuantError(Exception):
    ...
_type_traits: dict[GGMLQuantizationType, type[__Quant]] = {}
def quantize(data, qtype):
    if qtype == GGMLQuantizationType.F32:
        return data.astype(np.float32, copy=False)
    elif qtype == GGMLQuantizationType.F16:
        return data.astype(np.float16, copy=False)
    elif (q := _type_traits.get(qtype)) is not None:
        return q.quantize(data)
    else:
        raise NotImplementedError(
            f'Quantization for {qtype.name} is not yet implemented')
def dequantize(data, qtype):
    if qtype == GGMLQuantizationType.F32:
        return data.view(np.float32)
    elif qtype == GGMLQuantizationType.F16:
        return data.view(np.float16).astype(np.float32)
    elif (q := _type_traits.get(qtype)) is not None:
        return q.dequantize(data)
    else:
        raise NotImplementedError(
            f'Dequantization for {qtype.name} is not yet implemented')
class __Quant(ABC):
    qtype: GGMLQuantizationType
    block_size: int
    type_size: int
    grid: np.ndarray[Any, np.dtype[np.float32]] | None = None
    grid_shape: tuple[int, int] = (0, 0)
    grid_map: tuple[int | float, ...] = ()
    grid_hex: bytes | None = None
    def __init__(self):
        return TypeError("Quant conversion classes can't have instances")
    def __init_subclass__(cls, qtype):
        cls.qtype = qtype
        cls.block_size, cls.type_size = GGML_QUANT_SIZES[qtype]
        cls.__quantize_lazy = LazyNumpyTensor._wrap_fn(cls.__quantize_array,
            meta_noop=(np.uint8, cls.__shape_to_bytes))
        cls.__dequantize_lazy = LazyNumpyTensor._wrap_fn(cls.
            __dequantize_array, meta_noop=(np.float32, cls.__shape_from_bytes))
        assert qtype not in _type_traits
        _type_traits[qtype] = cls
    @classmethod
    def init_grid(cls):
        if cls.grid is not None or cls.grid_hex is None:
            return
        bits_per_elem = ceil(log2(len(cls.grid_map)))
        assert bits_per_elem != 0, cls.qtype.name
        elems_per_byte = 8 // bits_per_elem
        grid = np.frombuffer(cls.grid_hex, dtype=np.uint8)
        grid = grid.reshape((-1, 2))
        grid = (np.where(grid > 64, grid + 9, grid) & 15) << np.array([4, 0
            ], dtype=np.uint8).reshape((1, 2))
        grid = grid[..., 0] | grid[..., 1]
        grid = grid.reshape((-1, 1)) >> np.array([i for i in range(0, 8, 8 //
            elems_per_byte)], dtype=np.uint8).reshape((1, elems_per_byte))
        grid = (grid & (1 << bits_per_elem) - 1).reshape((-1, 1))
        grid_map = np.array(cls.grid_map, dtype=np.float32).reshape((1, -1))
        grid = np.take_along_axis(grid_map, grid, axis=-1)
        cls.grid = grid.reshape((1, 1, *cls.grid_shape))
    @classmethod
    @abstractmethod
    def quantize_blocks(cls, blocks):
        raise NotImplementedError
    @classmethod
    @abstractmethod
    def dequantize_blocks(cls, blocks):
        raise NotImplementedError
    @classmethod
    def quantize_rows(cls, rows):
        rows = rows.astype(np.float32, copy=False)
        shape = rows.shape
        n_blocks = rows.size // cls.block_size
        blocks = rows.reshape((n_blocks, cls.block_size))
        blocks = cls.quantize_blocks(blocks)
        assert blocks.dtype == np.uint8
        assert blocks.shape[-1] == cls.type_size
        return blocks.reshape(cls.__shape_to_bytes(shape))
    @classmethod
    def dequantize_rows(cls, rows):
        rows = rows.view(np.uint8)
        shape = rows.shape
        n_blocks = rows.size // cls.type_size
        blocks = rows.reshape((n_blocks, cls.type_size))
        blocks = cls.dequantize_blocks(blocks)
        assert blocks.dtype == np.float32
        assert blocks.shape[-1] == cls.block_size
        return blocks.reshape(cls.__shape_from_bytes(shape))
    @classmethod
    def __shape_to_bytes(cls, shape):
        return quant_shape_to_byte_shape(shape, cls.qtype)
    @classmethod
    def __shape_from_bytes(cls, shape):
        return quant_shape_from_byte_shape(shape, cls.qtype)
    @classmethod
    def __quantize_array(cls, array):
        return _apply_over_grouped_rows(cls.quantize_rows, arr=array, otype
            =np.uint8, oshape=cls.__shape_to_bytes(array.shape))
    @classmethod
    def __dequantize_array(cls, array):
        cls.init_grid()
        return _apply_over_grouped_rows(cls.dequantize_rows, arr=array,
            otype=np.float32, oshape=cls.__shape_from_bytes(array.shape))
    @classmethod
    def __quantize_lazy(cls, lazy_tensor: LazyNumpyTensor, /):
        pass
    @classmethod
    def __dequantize_lazy(cls, lazy_tensor: LazyNumpyTensor, /):
        pass
    @classmethod
    def can_quantize(cls, tensor):
        return tensor.shape[-1] % cls.block_size == 0
    @classmethod
    def quantize(cls, tensor):
        if not cls.can_quantize(tensor):
            raise QuantError(
                f"Can't quantize tensor with shape {tensor.shape} to {cls.qtype.name}"
                )
        if isinstance(tensor, LazyNumpyTensor):
            return cls.__quantize_lazy(tensor)
        else:
            return cls.__quantize_array(tensor)
    @classmethod
    def dequantize(cls, tensor):
        if isinstance(tensor, LazyNumpyTensor):
            return cls.__dequantize_lazy(tensor)
        else:
            return cls.__dequantize_array(tensor)
class BF16(__Quant, qtype=GGMLQuantizationType.BF16):
    @classmethod
    def quantize_blocks(cls, blocks):
        n = blocks.view(np.uint32)
        n = np.where(n & 2147483647 > 2139095040, n & np.uint32(4294901760) |
            np.uint32(64 << 16), n)
        n = np.uint64(n) + (32767 + (n >> 16 & 1)) >> 16
        return n.astype(np.uint16).view(np.uint8)
    @classmethod
    def dequantize_blocks(cls, blocks):
        return (blocks.view(np.int16).astype(np.int32) << 16).view(np.float32)
class Q4_0(__Quant, qtype=GGMLQuantizationType.Q4_0):
    @classmethod
    def quantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        imax = abs(blocks).argmax(axis=-1, keepdims=True)
        max = np.take_along_axis(blocks, imax, axis=-1)
        d = max / -8
        with np.errstate(divide='ignore'):
            id = np.where(d == 0, 0, 1 / d)
        qs = np.trunc(blocks * id + np.float32(8.5), dtype=np.float32).astype(
            np.uint8).clip(0, 15)
        qs = qs.reshape((n_blocks, 2, cls.block_size // 2))
        qs = qs[..., 0, :] | qs[..., 1, :] << np.uint8(4)
        d = d.astype(np.float16).view(np.uint8)
        return np.concatenate([d, qs], axis=-1)
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d, qs = np.hsplit(blocks, [2])
        d = d.view(np.float16).astype(np.float32)
        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([
            0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(15)).reshape((n_blocks, -1)).astype(np.int8
            ) - np.int8(8)
        return d * qs.astype(np.float32)
class Q4_1(__Quant, qtype=GGMLQuantizationType.Q4_1):
    @classmethod
    def quantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        max = blocks.max(axis=-1, keepdims=True)
        min = blocks.min(axis=-1, keepdims=True)
        d = (max - min) / 15
        with np.errstate(divide='ignore'):
            id = np.where(d == 0, 0, 1 / d)
        qs = np.trunc((blocks - min) * id + np.float32(0.5), dtype=np.float32
            ).astype(np.uint8).clip(0, 15)
        qs = qs.reshape((n_blocks, 2, cls.block_size // 2))
        qs = qs[..., 0, :] | qs[..., 1, :] << np.uint8(4)
        d = d.astype(np.float16).view(np.uint8)
        m = min.astype(np.float16).view(np.uint8)
        return np.concatenate([d, m, qs], axis=-1)
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d, rest = np.hsplit(blocks, [2])
        m, qs = np.hsplit(rest, [2])
        d = d.view(np.float16).astype(np.float32)
        m = m.view(np.float16).astype(np.float32)
        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([
            0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(15)).reshape((n_blocks, -1)).astype(np.float32)
        return d * qs + m
class Q5_0(__Quant, qtype=GGMLQuantizationType.Q5_0):
    @classmethod
    def quantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        imax = abs(blocks).argmax(axis=-1, keepdims=True)
        max = np.take_along_axis(blocks, imax, axis=-1)
        d = max / -16
        with np.errstate(divide='ignore'):
            id = np.where(d == 0, 0, 1 / d)
        q = np.trunc(blocks * id + np.float32(16.5), dtype=np.float32).astype(
            np.uint8).clip(0, 31)
        qs = q.reshape((n_blocks, 2, cls.block_size // 2))
        qs = qs[..., 0, :] & np.uint8(15) | qs[..., 1, :] << np.uint8(4)
        qh = np.packbits(q.reshape((n_blocks, 1, 32)) >> np.uint8(4), axis=
            -1, bitorder='little').reshape(n_blocks, 4)
        d = d.astype(np.float16).view(np.uint8)
        return np.concatenate([d, qh, qs], axis=-1)
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d, rest = np.hsplit(blocks, [2])
        qh, qs = np.hsplit(rest, [4])
        d = d.view(np.float16).astype(np.float32)
        qh = qh.view(np.uint32)
        qh = qh.reshape((n_blocks, 1)) >> np.array([i for i in range(32)],
            dtype=np.uint32).reshape((1, 32))
        ql = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([
            0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qh = (qh & np.uint32(1)).astype(np.uint8)
        ql = (ql & np.uint8(15)).reshape((n_blocks, -1))
        qs = (ql | qh << np.uint8(4)).astype(np.int8) - np.int8(16)
        return d * qs.astype(np.float32)
class Q5_1(__Quant, qtype=GGMLQuantizationType.Q5_1):
    @classmethod
    def quantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        max = blocks.max(axis=-1, keepdims=True)
        min = blocks.min(axis=-1, keepdims=True)
        d = (max - min) / 31
        with np.errstate(divide='ignore'):
            id = np.where(d == 0, 0, 1 / d)
        q = np.trunc((blocks - min) * id + np.float32(0.5), dtype=np.float32
            ).astype(np.uint8).clip(0, 31)
        qs = q.reshape((n_blocks, 2, cls.block_size // 2))
        qs = qs[..., 0, :] & np.uint8(15) | qs[..., 1, :] << np.uint8(4)
        qh = np.packbits(q.reshape((n_blocks, 1, 32)) >> np.uint8(4), axis=
            -1, bitorder='little').reshape(n_blocks, 4)
        d = d.astype(np.float16).view(np.uint8)
        m = min.astype(np.float16).view(np.uint8)
        return np.concatenate([d, m, qh, qs], axis=-1)
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d, rest = np.hsplit(blocks, [2])
        m, rest = np.hsplit(rest, [2])
        qh, qs = np.hsplit(rest, [4])
        d = d.view(np.float16).astype(np.float32)
        m = m.view(np.float16).astype(np.float32)
        qh = qh.view(np.uint32)
        qh = qh.reshape((n_blocks, 1)) >> np.array([i for i in range(32)],
            dtype=np.uint32).reshape((1, 32))
        ql = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([
            0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qh = (qh & np.uint32(1)).astype(np.uint8)
        ql = (ql & np.uint8(15)).reshape((n_blocks, -1))
        qs = (ql | qh << np.uint8(4)).astype(np.float32)
        return d * qs + m
class Q8_0(__Quant, qtype=GGMLQuantizationType.Q8_0):
    @classmethod
    def quantize_blocks(cls, blocks):
        d = abs(blocks).max(axis=1, keepdims=True) / 127
        with np.errstate(divide='ignore'):
            id = np.where(d == 0, 0, 1 / d)
        qs = np_roundf(blocks * id)
        d = d.astype(np.float16).view(np.uint8)
        qs = qs.astype(np.int8).view(np.uint8)
        return np.concatenate([d, qs], axis=1)
    @classmethod
    def dequantize_blocks(cls, blocks):
        d, x = np.split(blocks, [2], axis=1)
        d = d.view(np.float16).astype(np.float32)
        x = x.view(np.int8).astype(np.float32)
        return x * d
class Q2_K(__Quant, qtype=GGMLQuantizationType.Q2_K):
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        scales, rest = np.hsplit(blocks, [QK_K // 16])
        qs, rest = np.hsplit(rest, [QK_K // 4])
        d, dmin = np.hsplit(rest, [2])
        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)
        dl = (d * (scales & 15).astype(np.float32)).reshape((n_blocks, QK_K //
            16, 1))
        ml = (dmin * (scales >> 4).astype(np.float32)).reshape((n_blocks, 
            QK_K // 16, 1))
        shift = np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
        qs = qs.reshape((n_blocks, -1, 1, 32)) >> shift & np.uint8(3)
        qs = qs.reshape((n_blocks, QK_K // 16, 16)).astype(np.float32)
        qs = dl * qs - ml
        return qs.reshape((n_blocks, -1))
class Q3_K(__Quant, qtype=GGMLQuantizationType.Q3_K):
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        hmask, rest = np.hsplit(blocks, [QK_K // 8])
        qs, rest = np.hsplit(rest, [QK_K // 4])
        scales, d = np.hsplit(rest, [12])
        d = d.view(np.float16).astype(np.float32)
        lscales, hscales = np.hsplit(scales, [8])
        lscales = lscales.reshape((n_blocks, 1, 8)) >> np.array([0, 4],
            dtype=np.uint8).reshape((1, 2, 1))
        lscales = lscales.reshape((n_blocks, 16))
        hscales = hscales.reshape((n_blocks, 1, 4)) >> np.array([0, 2, 4, 6
            ], dtype=np.uint8).reshape((1, 4, 1))
        hscales = hscales.reshape((n_blocks, 16))
        scales = lscales & np.uint8(15) | (hscales & np.uint8(3)) << np.uint8(4
            )
        scales = (scales.astype(np.int8) - np.int8(32)).astype(np.float32)
        dl = (d * scales).reshape((n_blocks, 16, 1))
        ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6],
            dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = hmask.reshape(n_blocks, -1, 1, 32) >> np.array([i for i in
            range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
        ql = ql.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(3)
        qh = qh.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(1)
        qh = qh ^ np.uint8(1)
        q = (ql.astype(np.int8) - (qh << np.uint8(2)).astype(np.int8)).astype(
            np.float32)
        return (dl * q).reshape((n_blocks, QK_K))
class Q4_K(__Quant, qtype=GGMLQuantizationType.Q4_K):
    K_SCALE_SIZE = 12
    @staticmethod
    def get_scale_min(scales):
        n_blocks = scales.shape[0]
        scales = scales.view(np.uint8)
        scales = scales.reshape((n_blocks, 3, 4))
        d, m, m_d = np.split(scales, 3, axis=-2)
        sc = np.concatenate([d & 63, m_d & 15 | d >> 2 & 48], axis=-1)
        min = np.concatenate([m & 63, m_d >> 4 | m >> 2 & 48], axis=-1)
        return sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8))
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d, rest = np.hsplit(blocks, [2])
        dmin, rest = np.hsplit(rest, [2])
        scales, qs = np.hsplit(rest, [cls.K_SCALE_SIZE])
        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)
        sc, m = Q4_K.get_scale_min(scales)
        d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
        dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))
        qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np
            .uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(15)).reshape((n_blocks, -1, 32)).astype(np.float32)
        return (d * qs - dm).reshape((n_blocks, QK_K))
class Q5_K(__Quant, qtype=GGMLQuantizationType.Q5_K):
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d, rest = np.hsplit(blocks, [2])
        dmin, rest = np.hsplit(rest, [2])
        scales, rest = np.hsplit(rest, [Q4_K.K_SCALE_SIZE])
        qh, qs = np.hsplit(rest, [QK_K // 8])
        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)
        sc, m = Q4_K.get_scale_min(scales)
        d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
        dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))
        ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np
            .uint8).reshape((1, 1, 2, 1))
        qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([i for i in
            range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
        ql = (ql & np.uint8(15)).reshape((n_blocks, -1, 32))
        qh = (qh & np.uint8(1)).reshape((n_blocks, -1, 32))
        q = (ql | qh << np.uint8(4)).astype(np.float32)
        return (d * q - dm).reshape((n_blocks, QK_K))
class Q6_K(__Quant, qtype=GGMLQuantizationType.Q6_K):
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        ql, rest = np.hsplit(blocks, [QK_K // 2])
        qh, rest = np.hsplit(rest, [QK_K // 4])
        scales, d = np.hsplit(rest, [QK_K // 16])
        scales = scales.view(np.int8).astype(np.float32)
        d = d.view(np.float16).astype(np.float32)
        d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
        ql = ql.reshape((n_blocks, -1, 1, 64)) >> np.array([0, 4], dtype=np
            .uint8).reshape((1, 1, 2, 1))
        ql = (ql & np.uint8(15)).reshape((n_blocks, -1, 32))
        qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6],
            dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = (qh & np.uint8(3)).reshape((n_blocks, -1, 32))
        q = (ql | qh << np.uint8(4)).astype(np.int8) - np.int8(32)
        q = q.reshape((n_blocks, QK_K // 16, -1)).astype(np.float32)
        return (d * q).reshape((n_blocks, QK_K))
class TQ1_0(__Quant, qtype=GGMLQuantizationType.TQ1_0):
    @classmethod
    def quantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d = abs(blocks).max(axis=-1, keepdims=True)
        with np.errstate(divide='ignore'):
            id = np.where(d == 0, 0, 1 / d)
        qs = np_roundf(blocks * id)
        qs = (qs.astype(np.int8) + np.int8(1)).astype(np.uint8)
        qs0, qs1, qh = qs[..., :32 * 5], qs[..., 32 * 5:48 * 5], qs[..., 48 *
            5:]
        qs0 = qs0.reshape((n_blocks, -1, 5, 32)) * np.array([81, 27, 9, 3, 
            1], dtype=np.uint8).reshape((1, 1, 5, 1))
        qs0 = np.sum(qs0, axis=-2).reshape((n_blocks, -1))
        qs1 = qs1.reshape((n_blocks, -1, 5, 16)) * np.array([81, 27, 9, 3, 
            1], dtype=np.uint8).reshape((1, 1, 5, 1))
        qs1 = np.sum(qs1, axis=-2).reshape((n_blocks, -1))
        qh = qh.reshape((n_blocks, -1, 4, 4)) * np.array([81, 27, 9, 3],
            dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = np.sum(qh, axis=-2).reshape((n_blocks, -1))
        qs = np.concatenate([qs0, qs1, qh], axis=-1)
        qs = (qs.astype(np.uint16) * 256 + (243 - 1)) // 243
        qs = qs.astype(np.uint8)
        d = d.astype(np.float16).view(np.uint8)
        return np.concatenate([qs, d], axis=-1)
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        qs, rest = np.hsplit(blocks, [(QK_K - 4 * QK_K // 64) // 5])
        qh, d = np.hsplit(rest, [QK_K // 64])
        d = d.view(np.float16).astype(np.float32)
        qs0, qs1 = qs[..., :32], qs[..., 32:]
        qs0 = qs0.reshape((n_blocks, -1, 1, 32)) * np.array([1, 3, 9, 27, 
            81], dtype=np.uint8).reshape((1, 1, 5, 1))
        qs0 = qs0.reshape((n_blocks, -1))
        qs1 = qs1.reshape((n_blocks, -1, 1, 16)) * np.array([1, 3, 9, 27, 
            81], dtype=np.uint8).reshape((1, 1, 5, 1))
        qs1 = qs1.reshape((n_blocks, -1))
        qh = qh.reshape((n_blocks, -1, 1, 4)) * np.array([1, 3, 9, 27],
            dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = qh.reshape((n_blocks, -1))
        qs = np.concatenate([qs0, qs1, qh], axis=-1)
        qs = (qs.astype(np.uint16) * 3 >> 8).astype(np.int8) - np.int8(1)
        return d * qs.astype(np.float32)
class TQ2_0(__Quant, qtype=GGMLQuantizationType.TQ2_0):
    @classmethod
    def quantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d = abs(blocks).max(axis=-1, keepdims=True)
        with np.errstate(divide='ignore'):
            id = np.where(d == 0, 0, 1 / d)
        qs = np_roundf(blocks * id)
        qs = (qs.astype(np.int8) + np.int8(1)).astype(np.uint8)
        qs = qs.reshape((n_blocks, -1, 4, 32)) << np.array([0, 2, 4, 6],
            dtype=np.uint8).reshape((1, 1, 4, 1))
        qs = qs[..., 0, :] | qs[..., 1, :] | qs[..., 2, :] | qs[..., 3, :]
        qs = qs.reshape((n_blocks, -1))
        d = d.astype(np.float16).view(np.uint8)
        return np.concatenate([qs, d], axis=-1)
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        qs, d = np.hsplit(blocks, [QK_K // 4])
        d = d.view(np.float16).astype(np.float32)
        qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6],
            dtype=np.uint8).reshape((1, 1, 4, 1))
        qs = (qs & 3).reshape((n_blocks, -1)).astype(np.int8) - np.int8(1)
        return d * qs.astype(np.float32)
class MXFP4(__Quant, qtype=GGMLQuantizationType.MXFP4):
    kvalues = 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
    @staticmethod
    def e8m0_to_fp32_half(x):
        bits = np.where(x < 2, np.uint32(2097152) << np.uint32(x), np.
            uint32(x - 1) << np.uint32(23))
        return bits.view(np.float32)
    @classmethod
    def quantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d = abs(blocks).max(axis=-1, keepdims=True)
        with np.errstate(divide='ignore'):
            e = np.where(d > 0, np.floor(np.log2(d)) - 2 + 127, 0).astype(np
                .uint8)
        d = cls.e8m0_to_fp32_half(e)
        kvalues = np.array(cls.kvalues, dtype=np.int8).reshape((1, 1, 16))
        errs = np.abs(d.reshape((n_blocks, 1, 1)) * kvalues.astype(np.
            float32) - blocks.reshape((n_blocks, cls.block_size, 1)))
        best = np.argmin(errs, axis=-1, keepdims=True)
        qs = best.reshape(n_blocks, 2, cls.block_size // 2).astype(np.uint8)
        qs = qs[:, 0] | qs[:, 1] << np.uint8(4)
        qs = qs.reshape((n_blocks, cls.block_size // 2))
        return np.concatenate([e, qs], axis=-1)
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        e, qs = np.hsplit(blocks, [1])
        d = cls.e8m0_to_fp32_half(e)
        qs = qs.reshape((n_blocks, 1, cls.block_size // 2)) >> np.array([0,
            4], dtype=np.uint8).reshape((1, 2, 1))
        qs = (qs & np.uint8(15)).view(np.int8)
        kvalues = np.array(cls.kvalues, dtype=np.int8).reshape(1, 1, 16)
        qs = np.take_along_axis(kvalues, qs, axis=-1).reshape((n_blocks,
            cls.block_size))
        return d * qs.astype(np.float32)
class IQ2_XXS(__Quant, qtype=GGMLQuantizationType.IQ2_XXS):
    ksigns: bytes = (
        b'\x00\x81\x82\x03\x84\x05\x06\x87\x88\t\n\x8b\x0c\x8d\x8e\x0f\x90\x11\x12\x93\x14\x95\x96\x17\x18\x99\x9a\x1b\x9c\x1d\x1e\x9f\xa0!"\xa3$\xa5\xa6\'(\xa9\xaa+\xac-.\xaf0\xb1\xb23\xb456\xb7\xb89:\xbb<\xbd\xbe?\xc0AB\xc3D\xc5\xc6GH\xc9\xcaK\xccMN\xcfP\xd1\xd2S\xd4UV\xd7\xd8YZ\xdb\\\xdd\xde_`\xe1\xe2c\xe4ef\xe7\xe8ij\xebl\xed\xeeo\xf0qr\xf3t\xf5\xf6wx\xf9\xfa{\xfc}~\xff'
        )
    grid_shape = 256, 8
    grid_map = 8, 25, 43
    grid_hex = (
        b'00000200050008000a00110014002000220028002a00410044005000580061006400800082008a00a20001010401100115014001840198010002020222028202010404041004210424044004420448046004810484049004a404000502050805200546056905800591050906100640068406a406000805080808140828084108440850085208880804094009020a140a01100410101021104010601084109010951000110811201150115a1180112412451200140814201425144914801418156215001616160118041810184018811800190519a019511a002002200a2044206120802082202921482100220222012404241024402456240025412564259026082820289428442a0140044010401840214024404040484056406040814084409040004120416141804185410142104248425642684200440844204480449944124524450046014804481048404845480049584961498249454a904a005008501150195020508050885004514251a4519152905492540a550156545600581158195864584059085a04601060406068600061556118626062006405641065126584654268008002800a8041808280048118814081118201840484108415844084608400854685948509864086608602880489118a0490109024904090a19016918091459200942294449451958198209902a050a085a009a100a218a450a804a9'
        )
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d, qs = np.hsplit(blocks, [2])
        d = d.view(np.float16).astype(np.float32)
        qs = qs.view(np.uint32).reshape(n_blocks, -1, 2)
        db = d * (np.float32(0.5) + (qs[..., 1] >> 28).astype(np.float32)
            ) * np.float32(0.25)
        db = db.reshape((n_blocks, -1, 1, 1))
        signs = qs[..., 1].reshape((n_blocks, -1, 1)) >> np.array([0, 7, 14,
            21], dtype=np.uint32).reshape((1, 1, 4))
        ksigns = np.frombuffer(cls.ksigns, dtype=np.uint8).reshape((1, 1, 1,
            128))
        signs = (signs & np.uint32(127)).reshape((n_blocks, -1, 4, 1))
        signs = np.take_along_axis(ksigns, signs, axis=-1)
        signs = signs.reshape((n_blocks, -1, 4, 1)) >> np.array([i for i in
            range(8)], dtype=np.uint8).reshape((1, 1, 1, 8))
        signs = signs & np.uint8(1)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 4, 8))
        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs[..., 0].copy().view(np.uint8
            ).reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 4, 8))
        return (db * grid * signs).reshape((n_blocks, -1))
class IQ2_XS(__Quant, qtype=GGMLQuantizationType.IQ2_XS):
    grid_shape = 512, 8
    grid_map = 8, 25, 43
    grid_hex = (
        b'00000200050008000a001100140016001900200022002500280041004400460049005000520055005800610064008000820085008800910094009900a000010104010601090110011201150118011a0121012401400142014501480151015401600168018101840190010002020205020802110214022002410244025002550280028a02010404040604090410041204150418042104240440044204450448045104540456046004810484049004000502050505080511051405200541054405500561058005010604061006260640064206840600080208050808080a08110814082008250841084408500858088008a008aa08010904091009400981098909000a200a280a960aa00a011004100610091010101210151018102110241040104210451048105110541060106a108110841090100011021105110811111114112011411144115011801194119611011204120612101240126012001402140514081411141414201441144414491450146414801401150415101540150016141649160118041810181218401854188618001905196619511aa91a00200220052008200a201120142020204120442050208020a020012104211021402148216521002222228022a82201240424102429244024002541255225992501261a26a626002808280a28202855288828a22868299029082a202a822a882a8a2a014004400640094010401240154018402140244040404240454048404a405140544060406540814084409040004102410541084111411441204141414441504180418541a2410142044210421242294240420044024405440844114414441944204441444444504480449444014504451045244540459a4500460a4644465046014804481048404845485448624800491149444950496949044a00500250055008501150145020502850415044505050805001510451105115514051425100524452aa520154045410542154405460548154a154005508558055885521566856a156005814584158505899581a5940594259855a0160046010604060546062608660a960006124624a62926200641664106540654565a46501686a682569066a546a626a00800280058008801180148020802a8041804480508080808280a880aa8001810481068110814081518159810082208280828282a082a8820184048410841284158440846084898400854485a58518866a860088088825885a8880888288a8880689228a808a888a968aa88a019004901090409056908490009122916491569289920094059444945094589429959095929541965198a6984999159a609a00a002a008a00aa020a02aa0a0a051a159a1a6a100a202a208a22aa280a2a0a240a495a465a698a60aa820a822a828a8a0a8a8a804a984a986a928aa2aaa91aaaaaa'
        )
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d, rest = np.hsplit(blocks, [2])
        qs, scales = np.hsplit(rest, [2 * QK_K // 8])
        d = d.view(np.float16).astype(np.float32)
        qs = qs.view(np.uint16)
        scales = scales.reshape((n_blocks, -1, 1)) >> np.array([0, 4],
            dtype=np.uint8).reshape((1, 1, 2))
        scales = (scales & 15).reshape((n_blocks, -1))
        db = d * (np.float32(0.5) + scales) * np.float32(0.25)
        db = db.reshape((n_blocks, -1, 1, 1))
        signs = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).reshape(1, 1, 128
            )
        signs = np.take_along_axis(signs, (qs >> 9).reshape((n_blocks, -1, 
            1)), axis=-1)
        signs = signs.reshape((n_blocks, -1, 1)) >> np.array([i for i in
            range(8)], dtype=np.uint8).reshape((1, 1, 8))
        signs = signs & np.uint8(1)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 2, 8))
        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, (qs & np.uint16(511)).reshape((
            n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 2, 8))
        return (db * grid * signs).reshape((n_blocks, -1))
class IQ2_S(__Quant, qtype=GGMLQuantizationType.IQ2_S):
    grid_shape = 1024, 8
    grid_map = 8, 25, 43
    grid_hex = (
        b'00000200050008000a0011001400160019002000220025002800410044004600490050005200550058006100640066006900800082008500880091009400a000a500aa0001010401060109011001120115011801210124014001420145014801510154015601590160016501680181018401900192019501a101a40100020202050208021102140220022a02410244024602490250025502800285028a029402a20201040404060409041004120415041804210424042604290440044204450448044a0451045404560459046004620465048104840486048904900495049804a104a40400050205050508050a0511051405160519052005250528054105440546054905500552055505580561056405800582058505880591059405a00501060406060609061006150640064506480651065406600681068406900600080208050808081108140816081908200825082a084108440846084908500852085508580861086408800885089408aa08010904091009120915091809210940094509480951095409600981099009000a110a140a220a280a2a0a500a990a01100410061009101010121015101810211024102610401042104510481051105410561059106010621065106810811084108610901095109810a110a41000110211051108110a1111111411161119112011221125112811411144114611491150115211551158116111641180118211851188119111941101120412091210121512211224124012451251125412811284129012001402140514081411141414161419142014251428144114441446144914501452145514581461146414801482148514881491149414a014011504150615091510151215151518152115241540154215451548155115541560158115841590150016051608161116141620164116441650168016aa160118041806180918101815181818211840184218451848185118541860188118841800190219051908191119141920194119441950196919a219041a101a401a561a00200220052008201120142016201920202025202a2041204420502052205520642080208a209420aa2001210421102112211521212140214221452151215421602181218421902100220a22222228222a224422502288228a22a8220124042406240924102415241824212424244024422445244824512454246024812484249024002505250825112514252025412544255025662580250126042610264026592600280528112814284128442850288a28aa2801290429102995290a2a222a642a882a8a2a014004400640094010401240154018401a4021402440264040404240454048404a40514054405640594060406240654081408440904095409840a140a440004102410541084111411441164119412041224125414141444146414941504152415541584161416441804182418541884191419441a04101420442104212421542184224424042454248425142544260428142844200440244054408440a44114414441644194420442244254428444144444446444944504452445544584461446444804482448544884491449444a044014504450645094510451245154518452145244540454245454548455145544560456a4581458445904500460246054608461146144620464146444650468046a546014804480948104812481548184821482448404842484548484851485448604884489048004902490549084911491449204941494449504980499649014a044a104a404a0050025005500850115014501650195020502250255028504150445046504950505052505550585061506450805082508550885091509450015104510651095110511251155118512151245140514251455148515151545160518151845190510052055208521152145220524152445250526952805201540454065409541054125415541854215424544054425445544854515454546054815484549054005502550555085511551455205541554455505580550156045610562656405600580258055808581158145820584158445850585a5880580159045910594059005a195a855aa85a016004600660106012601560186021602460406045604860516054606060846090600061026105610861116114612061416144615061806199610462106240625662a162006405640864116414642064416444645064806401650465106540654a6568659265006694660168046810686568986800692a69426aa16a00800280058008801180148019802080258041804480508052805580588061808080858091809480018104810981108112811581188121812481408142814581488151815481818184819081a981008205820a82118214824182448250820184048406840984108412841584188421844084428445844884518454846084818484849084008502850585088511851485208541854485508580858a85018604861086298640860088058811881488418844885088a2880189048940896589228a588a5a8a828aa28a01900490099010901290159018902490409042904590489051905490609081908490909000910591119114914191449150915a910192049210924092a69200940294059408941194149420944194449450948094969401950495109540959895a19500964696649601980498109826984098a998009949995299909a00a005a00aa014a022a02aa041a044a050a0a2a0aaa040a165a102a20aa222a228a22aa282a288a28aa2a8a201a404a410a440a489a4a4a400a519a551a60aa828a8a2a854a986a908aa0aaa20aa22aa28aa88aaaaaa'
        )
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d, rest = np.hsplit(blocks, [2])
        qs, rest = np.hsplit(rest, [QK_K // 8])
        signs, rest = np.hsplit(rest, [QK_K // 8])
        qh, scales = np.hsplit(rest, [QK_K // 32])
        d = d.view(np.float16).astype(np.float32)
        scales = scales.reshape((n_blocks, -1, 1)) >> np.array([0, 4],
            dtype=np.uint8).reshape((1, 1, 2))
        scales = (scales & 15).reshape((n_blocks, -1))
        db = d * (np.float32(0.5) + scales) * np.float32(0.25)
        db = db.reshape((n_blocks, -1, 1, 1))
        signs = signs.reshape((n_blocks, -1, 1)) >> np.array([i for i in
            range(8)], dtype=np.uint8).reshape((1, 1, 8))
        signs = signs & np.uint8(1)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 2, 8))
        qh = qh.reshape((n_blocks, -1, 1)) >> np.array([0, 2, 4, 6], dtype=
            np.uint8).reshape((1, 1, 4))
        qs = qs.astype(np.uint16) | ((qh & 3).astype(np.uint16) << 8).reshape((
            n_blocks, -1))
        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)
            ), axis=-2)
        grid = grid.reshape((n_blocks, -1, 2, 8))
        return (db * grid * signs).reshape((n_blocks, -1))
class IQ3_XXS(__Quant, qtype=GGMLQuantizationType.IQ3_XXS):
    grid_shape = 256, 4
    grid_map = 4, 12, 20, 28, 36, 44, 52, 62
    grid_hex = (
        b'0000020004001100130017002000220031004200730075000101030110011201210125013001320141015401700100020202040211022002220231023302370251025702750201030703100312032503700313043704440457047304750401050705320552053506640610071407160743076107011003101010121021102310301032103410471050100011021111112011221101120312101212122112301272120013021320133113461366130114051450142015241546157115051622174017002002201120132020202220262031204220012103210521102112212121302163216721702100220222112217222022222237224022552201231023142370237423352453240325272541257425012703271627452701301030123021302330503065307230003102312031313144314631013203321032253252327232113333333034473472340035063522355535143636366336333760370440174035403740534057407441204237424042604266420743454304445144644425454345704505471047124730471250415070500051065126515551145232527252025353531054235427547254025531555056245742572460446046606460216161611762646230633663446405655265336603672167037005700770107032705270267140711272457252720073157333736073217441740075027524753076'
        )
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d, rest = np.hsplit(blocks, [2])
        qs, scales = np.hsplit(rest, [QK_K // 4])
        d = d.view(np.float16).astype(np.float32)
        scales = scales.view(np.uint32)
        db = d * (np.float32(0.5) + (scales >> 28).astype(np.float32)
            ) * np.float32(0.5)
        db = db.reshape((n_blocks, -1, 1, 1))
        signs = scales.reshape((n_blocks, -1, 1)) >> np.array([0, 7, 14, 21
            ], dtype=np.uint32).reshape((1, 1, 4))
        ksigns = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).reshape((1, 
            1, 1, 128))
        signs = (signs & np.uint32(127)).reshape((n_blocks, -1, 4, 1))
        signs = np.take_along_axis(ksigns, signs, axis=-1)
        signs = signs.reshape((n_blocks, -1, 4, 1)) >> np.array([i for i in
            range(8)], dtype=np.uint8).reshape((1, 1, 1, 8))
        signs = signs & np.uint8(1)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 4, 8))
        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)
            ), axis=-2)
        grid = grid.reshape((n_blocks, -1, 4, 8))
        return (db * grid * signs).reshape((n_blocks, -1))
class IQ3_S(__Quant, qtype=GGMLQuantizationType.IQ3_S):
    grid_shape = 512, 4
    grid_map = 1, 3, 5, 7, 9, 11, 13, 15
    grid_hex = (
        b'00000100020005000700100011001200140016002000210025003300400042004500470051005300600062007100740077000001010102010401100111011501200123012701310135014401610165017201000201020502070210021302160221022502300234024202450247025102530270027302030311031503200322033103330336034403500352036703710375030004130417042104240432044004430451047004020504052005220526053305410545054705660573050606110613063106520671060007020704072007220726073307500754070010011002100410101011101310151017102010221031103410361054105610611072100011011103110611101114112111301133114111501152117011761100121212151217122012241232124012431255126012721201130413071310131313211327133013341341136213701303140514121414143114331442144614501454140115101513152115301532155115201624162716441646160117031710171217211735174117621770170020012003200520072010201220142016202120232027203020322041204320452050205220672070207320752000210221102113211721222125213121342142215121012204220722212223223022372241225322572271227422002302230523112322232423312333234223502366230124072420242324322435244124722475240425112522253725402553257025002602260726212655266126052711272627302743275027023011301330153017302230313033303530423044304730513063307130013103310531143121312331403160317231763100321232203232323432503201331033143321332333273330334133433347335533733303341134163422343134523460346434013510351235253532354435563573351636413601370337203722373537004004401240204024402740324041405040704002410741114113412241304135414341514155410142034210421542214233424042574262427042044311431343204322433143354300440244244437444044714405450745214562451346344660461047154730474347514702501050145022504050445047505250665074500151035105511251215132517251005211522352305236525352025307531053275344535153655373530154045420543254465412552655515553554256025704572257116013601560316033606060006120612761646112623462426255626262706200631463216340632564436462640065036534656065056640661167136700700470077020702270367040705470627002711171247143714571017204721072167221723072517202733273357353730174057413742074507422754275027631760077'
        )
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d, rest = np.hsplit(blocks, [2])
        qs, rest = np.hsplit(rest, [QK_K // 4])
        qh, rest = np.hsplit(rest, [QK_K // 32])
        signs, scales = np.hsplit(rest, [QK_K // 8])
        d = d.view(np.float16).astype(np.float32)
        scales = scales.reshape((n_blocks, -1, 1)) >> np.array([0, 4],
            dtype=np.uint8).reshape((1, 1, 2))
        scales = (scales & 15).reshape((n_blocks, -1))
        db = d * (1 + 2 * scales)
        db = db.reshape((n_blocks, -1, 1, 1))
        signs = signs.reshape((n_blocks, -1, 1)) >> np.array([i for i in
            range(8)], dtype=np.uint8).reshape((1, 1, 8))
        signs = signs & np.uint8(1)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 4, 8))
        qh = qh.reshape((n_blocks, -1, 1)) >> np.array([i for i in range(8)
            ], dtype=np.uint8)
        qh = (qh & 1).astype(np.uint16).reshape((n_blocks, -1))
        qs = qs.astype(np.uint16) | qh << 8
        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)
            ), axis=-2)
        grid = grid.reshape((n_blocks, -1, 4, 8))
        return (db * grid * signs).reshape((n_blocks, -1))
class IQ1_S(__Quant, qtype=GGMLQuantizationType.IQ1_S):
    grid_shape = 2048, 8
    grid_map = -1, 0, 1
    grid_hex = (
        b'00000200050008000a00110015002000220028002a00450051005400560065008000820088008a009500a000a200a800aa000401050111011401160119011a012501410146014901520155015a0161016401660168018501910194019601a5010002020208020a0215022002220228022a02450251025902640269028002820288028a02910295029902a002a202a802aa0211041404160425044104490455045a046404650491049904a5040105040505050605150518051a052905400545054a0550055105540555055605590560056205650568056a0581059105950598059a05a105a405a505a605a90514061906410644065006520655065806600661066606690685069106940699060008020808080a0815082008220828082a0845085108560865088008820888088a089508a008a208a808aa0805091109140919092409250941095009510955096109640969099109940996099909a509000a020a080a0a0a150a200a220a280a2a0a450a510a590a610a650a800a820a850a880a8a0a950aa00aa20aa80aaa0a101011101410191024102510411044105010551058106110641065106910911094109610a110a5100111041106110911101112111511181121112411291145114a11501151115211541155115611591160116511841192119511a111a41111121412161225124012461249125212551258125a12641266128512911294129612a51201140614091414141514181419142114261441144514461448144a145114541455145614591462146514681484148914901494149514981499149a14a114a414a514a914021505150a151115141515151615191520152215251528152a1541154415451546155115521554155515561559155a1561156415651566156915801582158415851588158a159015911594159515961599159a15a015a215a51501160416051606161516161618161a1621162616401642164416451648164a1651165516561658165916611664166516681669166a1686168a1692169516a416a91611181618251841184418461849185018551858185a1860186118641866186918851891189418a5181019121915191a19211925194219441945194819511954195519561959195a19601965196a1989199119921995199819a119a619a919091a161a241a261a441a461a491a501a521a551a581a611a661a691a851a911a961a9a1a0020022008200a20152020202220252028202a20452051205920612065208020822088208a209520a020a220a520a820aa2005211121142119212521422144214921552158215a2161216421652166218521902196219921a521012208220a22112215222022222228222a2245225122562259226522812288228a2291229522a022a222a822aa220524142416241924252444244524462449245224552458245a2466248524912494249924a124a52409251525212529254025452548255125542555255925622565256825892590259425952598259a25a125a425a625a925052610261226192625264126492655266026612669268426862690269a260028022808280a2815282028222828282a2845285128542865288028822888288a28a028a228a828aa2809291129142919292529462949295229552961296429662969298529902996299929a429a529002a022a082a0a2a202a222a282a2a2a452a512a562a592a652a802a822a882a8a2a952aa02aa22aa82aaa2a054011401640254049405240554058405a4061406440664094409940a140a6400041014104410641094112411541164118411a41214126412941454148414a41514154415541564159415a41654168416a41814184418641904192419541a041a141a241054211421442164225424142524255425a426442694289429442a5420144154419442944454448444a44514454445544564461446244654468446a44814486448944904492449544a044a144a9440145024505450a4511451445154516451945204525452a45414544454545464549455045514554455545564558455945614564456545664569458245844585458845914594459545964599459a45a545a845aa450146054609461446154618461a462146244629464046424645464846504651465246554656465946624665466846814685468a4694469546a146a446a6460548114815481a48254842484948504855485848614864486648694885489148944896489948a5480149054906490a491049144915491849214924492649404945494a4951495249544955495649594960496249654966496a49864989499249954996499849a149a449a649a949164a444a464a494a554a584a5a4a644a694a944aa54a01500450055006500950125015501a5021502450295040504550485051505450555056505950655068508650895095509850a050a150a650a9500551085109510a5111511451155116511851195120512551265128512a5141514451455146514951505151515251545155515651585159515a51615164516551665169518251855191519451955196519951a051a551aa5101520652125215521a5221522452425245524a525152545255525652595262526552855290529252955299529a52a452045405541154145415541654185419542154255428542a54415444544554465449544a5450545154545455545654585459545a54615462546454655466546954805488548a5491549454955496549954a154a454a554aa5401550255045505550655095510551155125514551555165519551a552155245525552655295540554155425544554555465548554955505551555255545555555655585559555a5560556155645565556655685569556a5581558455855589558a559055915594559555965598559955a155a455a555a655a95500560156025604560656085609561156145615561856195620562156225624562556265628562956415645564656485649564a56505651565256545655565656585659565a566156645665566956825685568656885689568a56915695569a56a256a556a656a856a956045805580658095810581558185821582a58455848584a585158545855585658585859586058625864586558825889589058925895589858a158a9580159025905590a5911591459155916591959255941594459455946594959505951595259545955595659585959595a5961596459655966596959815985598959915994599559965998599959a559045a085a155a1a5a205a255a265a295a455a485a495a515a555a565a585a595a625a655a685a6a5a815a8a5a925a955a965a985a9a5aa15a05601460166019602560446050605560566058605a60616064606660696081609660a5600161046106610961126115612161226126612961456149615161556156615961656166616a6184618a6192619561a161a661a9611162166219624062416246625562566258626062856291629662a56211641264156416641a6421642664296440644264456448644a64516454645564566459645a646064626465648464856489649064926494649564966498649a64a164a464a964056508650a651165156516651965446545654665496550655165546555655665596561656465656566656965866589658a6591659565966599659a65a265a565a665a86502660966156620662666286629664066456648664a66516654665566566658665a666066656668668066826685668a669466966698669966a066a466a666aa661668196825684168526855685a6861686968856891689868a668016904691069156921692469266929694069416945694669486951695469556956695969606965696a69826984698a699569a169a469a569a969116a166a186a416a446a496a506a556a586a5a6a646a656a696a866a946a986a9a6aa66a0080028008800a802080228028802a8045805080518054805680598065808080828088808a809580a080a280a880aa8005811181148116811981258141814481498150815281558156815881598164816681698185818981948196819981a5810082028208820a8215822082228228822a8251825482598265828082828288828a829582a082a282a882aa821484198441844484518455845a846184648469849484998401850985128515851a85268529854085418545854885518554855585568559855a856585668568856a8581858485868589859085928595859885a68511861686198625864186448649864a865086558659865a86618666866a86858691869a86a4860088028808880a8815882088228828882a8841884588518854885988658869888088828888888a889588a088a288a888aa8805890689118914891689258941894489468949895089528955895a8961896489858996899989a589008a028a088a0a8a158a208a228a288a2a8a458a518a548a568a808a828a888a8a8a958aa08aa28aa88aaa8a059011901690189019902590419046904990559058905a9069906a9085909190949096909990a59001910491069109911091159118911a912191249126912991409145915091519154915591569159916291659184918691929195919891a191a491a691a99105921192149219922592449246924992509252925592589266926992859294929692a992019404940694109415941894269440944a945194549455945694589459946094619462946594849486949294949495949894a194a9940095059508950a951095119514951595169519952195259529952a9541954495459546954995509551955295549555955695589559955a95619564956595669569958195859588959195929594959595969599959a95a095a295a595a895aa950196049610961596199620962696299645964896499651965296559656965996659668968296849689968a96929694969596a496a696a99605981698199825984198469850985298559856985a98649865988598919896989998a59804990699099910991299159918991a99209921992499269940994299459948994a99519954995599569959996299659966996a99819984999099929995999a99a199a699059a159a259a449a469a499a509a559a589a619a859a919a949a959a969a00a002a008a00aa015a020a022a028a02aa045a051a054a056a059a080a082a088a08aa095a0a0a0a2a0a8a0aaa005a109a111a114a116a119a11aa146a149a151a155a158a15aa161a164a185a190a192a196a199a102a208a20aa210a219a222a228a22aa245a251a256a259a265a280a282a288a28aa295a2a0a2a2a2a8a2aaa219a425a441a444a450a454a455a458a45aa461a465a466a468a469a485a406a509a510a512a515a518a526a529a542a545a551a554a555a556a559a565a56aa581a584a585a586a589a592a595a598a505a611a616a61aa621a625a644a646a64aa652a655a656a658a660a662a686a690a695a696a699a6a1a6a4a6a6a600a802a808a80aa820a822a828a82aa851a854a856a859a880a882a888a88aa895a8a0a8a2a8a8a8aaa805a914a919a921a925a941a950a955a95aa961a966a969a990a996a900aa02aa08aa0aaa20aa22aa28aa2aaa51aa54aa56aa80aa82aa88aa8aaa95aaa0aaa2aaa8aaaaaa'
        )
    delta = np.float32(0.125)
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d, rest = np.hsplit(blocks, [2])
        qs, qh = np.hsplit(rest, [QK_K // 8])
        d = d.view(np.float16).astype(np.float32)
        qh = qh.view(np.uint16)
        dl = d * (2 * (qh >> 12 & 7) + 1)
        dl = dl.reshape((n_blocks, -1, 1, 1))
        delta = np.where(qh & np.uint16(32768) == 0, cls.delta, -cls.delta)
        delta = delta.reshape((n_blocks, -1, 1, 1))
        qh = qh.reshape((n_blocks, -1, 1)) >> np.array([0, 3, 6, 9], dtype=
            np.uint16).reshape((1, 1, 4))
        qs = qs.astype(np.uint16) | ((qh & 7) << 8).reshape((n_blocks, -1))
        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)
            ), axis=-2)
        grid = grid.reshape((n_blocks, -1, 4, 8))
        return (dl * (grid + delta)).reshape((n_blocks, -1))
class IQ1_M(__Quant, qtype=GGMLQuantizationType.IQ1_M):
    grid_shape = IQ1_S.grid_shape
    grid_map = IQ1_S.grid_map
    grid_hex = IQ1_S.grid_hex
    delta = IQ1_S.delta
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        qs, rest = np.hsplit(blocks, [QK_K // 8])
        qh, scales = np.hsplit(rest, [QK_K // 16])
        scales = scales.view(np.uint16)
        d = (scales.reshape((n_blocks, 4)) & np.uint16(61440)) >> np.array([
            12, 8, 4, 0], dtype=np.uint16).reshape((1, 4))
        d = d[..., 0] | d[..., 1] | d[..., 2] | d[..., 3]
        d = d.view(np.float16).astype(np.float32).reshape((n_blocks, 1))
        scales = scales.reshape(n_blocks, -1, 1) >> np.array([0, 3, 6, 9],
            dtype=np.uint16).reshape((1, 1, 4))
        scales = (scales & 7).reshape((n_blocks, -1))
        dl = d * (2 * scales + 1)
        dl = dl.reshape((n_blocks, -1, 2, 1, 1))
        qh = qh.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8
            ).reshape((1, 1, 2))
        qs = qs.astype(np.uint16) | ((qh & 7).astype(np.uint16) << 8).reshape((
            n_blocks, -1))
        delta = np.where(qh & 8 == 0, cls.delta, -cls.delta)
        delta = delta.reshape((n_blocks, -1, 2, 2, 1))
        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)
            ), axis=-2)
        grid = grid.reshape((n_blocks, -1, 2, 2, 8))
        return (dl * (grid + delta)).reshape((n_blocks, -1))
class IQ4_NL(__Quant, qtype=GGMLQuantizationType.IQ4_NL):
    kvalues = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53,
        69, 89, 113)
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d, qs = np.hsplit(blocks, [2])
        d = d.view(np.float16).astype(np.float32)
        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([
            0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(15)).reshape((n_blocks, -1, 1))
        kvalues = np.array(cls.kvalues, dtype=np.int8).reshape(1, 1, 16)
        qs = np.take_along_axis(kvalues, qs, axis=-1).astype(np.float32
            ).reshape((n_blocks, -1))
        return d * qs
class IQ4_XS(__Quant, qtype=GGMLQuantizationType.IQ4_XS):
    @classmethod
    def dequantize_blocks(cls, blocks):
        n_blocks = blocks.shape[0]
        d, rest = np.hsplit(blocks, [2])
        scales_h, rest = np.hsplit(rest, [2])
        scales_l, qs = np.hsplit(rest, [QK_K // 64])
        d = d.view(np.float16).astype(np.float32)
        scales_h = scales_h.view(np.uint16)
        scales_l = scales_l.reshape((n_blocks, -1, 1)) >> np.array([0, 4],
            dtype=np.uint8).reshape((1, 1, 2))
        scales_h = scales_h.reshape((n_blocks, 1, -1)) >> np.array([(2 * i) for
            i in range(QK_K // 32)], dtype=np.uint16).reshape((1, -1, 1))
        scales_l = scales_l.reshape((n_blocks, -1)) & np.uint8(15)
        scales_h = scales_h.reshape((n_blocks, -1)).astype(np.uint8
            ) & np.uint8(3)
        scales = (scales_l | scales_h << np.uint8(4)).astype(np.int8
            ) - np.int8(32)
        dl = (d * scales.astype(np.float32)).reshape((n_blocks, -1, 1))
        qs = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np
            .uint8).reshape((1, 1, 2, 1))
        qs = qs.reshape((n_blocks, -1, 32, 1)) & np.uint8(15)
        kvalues = np.array(IQ4_NL.kvalues, dtype=np.int8).reshape((1, 1, 1, -1)
            )
        qs = np.take_along_axis(kvalues, qs, axis=-1).astype(np.float32
            ).reshape((n_blocks, -1, 32))
        return (dl * qs).reshape((n_blocks, -1))