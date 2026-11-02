"""Typed Arrow codec for the message dataclasses.

Each message class gets an explicit Arrow ``struct`` type derived once from
its annotations (int -> int64, float -> float64, str -> utf8, bool -> bool,
IntEnum -> int32, nested dataclass -> struct, ``F64Array`` -> list<float64>,
...). A message travels as a one-row ``StructArray``; the schema name and
version go into the Dora metadata so receivers can reject mismatches.

The codec is pure (no Dora import) so nodes and tests can use it directly,
and the recorder can map the same schema to MCAP (JSON encoding).
"""
from __future__ import annotations

import dataclasses
import functools
import json
import typing
from enum import IntEnum
from typing import Any, Dict, Type, TypeVar

import numpy as np
import pyarrow as pa


SCHEMA_VERSION = 1
T = TypeVar("T")

_ARRAY_DTYPES = {"float64": (pa.float64(), np.float64), "int8": (pa.int8(), np.int8),
                 "uint8": (pa.uint8(), np.uint8)}


def _resolve_hints(cls) -> Dict[str, Any]:
    return typing.get_type_hints(cls, include_extras=True)


def _arrow_type(tp) -> pa.DataType:
    if typing.get_origin(tp) is typing.Annotated:
        _, tag = typing.get_args(tp)
        return pa.list_(_ARRAY_DTYPES[tag][0])
    if dataclasses.is_dataclass(tp):
        return struct_type(tp)
    if isinstance(tp, type) and issubclass(tp, IntEnum):
        return pa.int32()
    if tp is bool:
        return pa.bool_()
    if tp is int:
        return pa.int64()
    if tp is float:
        return pa.float64()
    if tp is str:
        return pa.string()
    raise TypeError(f"unsupported field type {tp!r}")


@functools.lru_cache(maxsize=None)
def struct_type(cls) -> pa.StructType:
    hints = _resolve_hints(cls)
    return pa.struct([pa.field(f.name, _arrow_type(hints[f.name]), nullable=False)
                      for f in dataclasses.fields(cls)])


def _to_py(obj) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_py(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, IntEnum):
        return int(obj)
    return obj


def encode(msg) -> pa.StructArray:
    """Message dataclass -> one-row Arrow StructArray with the declared schema."""
    return pa.array([_to_py(msg)], type=struct_type(type(msg)))


def metadata(msg) -> Dict[str, Any]:
    return {"schema": type(msg).__name__, "schema_version": SCHEMA_VERSION}


def _from_py(cls, value: Dict[str, Any]):
    hints = _resolve_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        tp, v = hints[f.name], value[f.name]
        if typing.get_origin(tp) is typing.Annotated:
            kwargs[f.name] = np.asarray(v, dtype=_ARRAY_DTYPES[typing.get_args(tp)[1]][1])
        elif dataclasses.is_dataclass(tp):
            kwargs[f.name] = _from_py(tp, v)
        elif isinstance(tp, type) and issubclass(tp, IntEnum):
            kwargs[f.name] = tp(v)
        else:
            kwargs[f.name] = v
    return cls(**kwargs)


def _struct_row_to_py(arr: pa.StructArray, cls) -> Dict[str, Any]:
    """Fast path: list fields are converted with ``to_numpy`` (no Python lists)."""
    hints = _resolve_hints(cls)
    out = {}
    for f in dataclasses.fields(cls):
        tp = hints[f.name]
        col = arr.field(f.name)
        if typing.get_origin(tp) is typing.Annotated:
            out[f.name] = col.flatten().to_numpy(zero_copy_only=False).copy()
        elif dataclasses.is_dataclass(tp):
            out[f.name] = _struct_row_to_py(col, tp)
        else:
            out[f.name] = col[0].as_py()
    return out


def decode(value: pa.Array, cls: Type[T]) -> T:
    """One-row Arrow StructArray -> message dataclass ``cls``."""
    if not isinstance(value, pa.StructArray) or len(value) != 1:
        raise ValueError(f"expected a one-row StructArray for {cls.__name__}, got {type(value).__name__}"
                         f" of length {len(value)}")
    if value.type != struct_type(cls):
        raise TypeError(f"schema mismatch for {cls.__name__}")
    return _from_py(cls, _struct_row_to_py(value, cls))


def check_metadata(meta: Dict[str, Any], cls) -> None:
    name = meta.get("schema")
    if name is not None and name != cls.__name__:
        raise TypeError(f"expected {cls.__name__}, got {name}")
    ver = meta.get("schema_version")
    if ver is not None and int(ver) != SCHEMA_VERSION:
        raise TypeError(f"{cls.__name__}: schema version {ver} != {SCHEMA_VERSION}")


# --------------------------------------------------------------- JSON (MCAP)
def to_json_dict(msg) -> Dict[str, Any]:
    """JSON-serialisable dict (arrays -> lists, enums -> ints)."""
    def conv(o):
        if dataclasses.is_dataclass(o):
            return {f.name: conv(getattr(o, f.name)) for f in dataclasses.fields(o)}
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, IntEnum):
            return int(o)
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o
    return conv(msg)


def from_json_dict(cls: Type[T], d: Dict[str, Any]) -> T:
    """Inverse of :func:`to_json_dict` (MCAP JSON -> message); null floats -> NaN."""
    hints = _resolve_hints(cls)
    fixed = {}
    for f in dataclasses.fields(cls):
        tp, v = hints[f.name], d[f.name]
        if dataclasses.is_dataclass(tp):
            fixed[f.name] = v
        elif tp is float and v is None:
            fixed[f.name] = float("nan")
        else:
            fixed[f.name] = v
    obj = _from_py(cls, {k: v for k, v in fixed.items()})
    for f in dataclasses.fields(cls):
        if dataclasses.is_dataclass(hints[f.name]):
            setattr(obj, f.name, from_json_dict(hints[f.name], d[f.name]))
    return obj


def json_schema(cls) -> Dict[str, Any]:
    """JSON Schema (draft 2020-12) of a message class, for MCAP channels."""
    hints = _resolve_hints(cls)
    props = {}
    for f in dataclasses.fields(cls):
        tp = hints[f.name]
        if typing.get_origin(tp) is typing.Annotated:
            item = "number" if typing.get_args(tp)[1] == "float64" else "integer"
            props[f.name] = {"type": "array", "items": {"type": item}}
        elif dataclasses.is_dataclass(tp):
            props[f.name] = json_schema(tp)
        elif tp is bool:
            props[f.name] = {"type": "boolean"}
        elif tp is str:
            props[f.name] = {"type": "string"}
        elif tp is float:
            props[f.name] = {"type": ["number", "null"]}
        else:
            props[f.name] = {"type": "integer"}
    return {"type": "object", "title": cls.__name__, "properties": props}


def dumps_json(msg) -> bytes:
    return json.dumps(to_json_dict(msg), separators=(",", ":")).encode()
