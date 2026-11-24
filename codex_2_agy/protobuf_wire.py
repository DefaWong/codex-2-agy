"""Zero-dependency Protocol Buffers binary wire encoder and decoder.

Implements standard protobuf wire types (varint, 64-bit, length-delimited, 32-bit)
plus helpers for google.protobuf.Timestamp, google.protobuf.Duration, and
google.protobuf.Any.
"""

from __future__ import annotations

import struct
from typing import Iterable, List, Tuple

WIRE_VARINT = 0
WIRE_FIXED64 = 1
WIRE_BYTES = 2
WIRE_FIXED32 = 5


def encode_varint(value: int) -> bytes:
    """Encode a signed/unsigned 64-bit integer as protobuf varint (two's complement for negatives)."""
    if value < 0:
        value &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = value & 0x7F
        value >>= 7
        if value:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def encode_tag(field_number: int, wire_type: int) -> bytes:
    return encode_varint((field_number << 3) | wire_type)


def f_varint(field_number: int, value: int | None, *, omit_zero: bool = True) -> bytes:
    if value is None:
        return b""
    if omit_zero and value == 0:
        return b""
    return encode_tag(field_number, WIRE_VARINT) + encode_varint(int(value))


def f_bool(field_number: int, value: bool | None, *, omit_false: bool = True) -> bytes:
    if value is None:
        return b""
    if omit_false and not value:
        return b""
    return encode_tag(field_number, WIRE_VARINT) + (b"\x01" if value else b"\x00")


def f_bytes(field_number: int, value: bytes | None, *, omit_empty: bool = True) -> bytes:
    if value is None:
        return b""
    if omit_empty and len(value) == 0:
        return b""
    return encode_tag(field_number, WIRE_BYTES) + encode_varint(len(value)) + value


def f_str(field_number: int, value: str | None, *, omit_empty: bool = True) -> bytes:
    if value is None:
        return b""
    if omit_empty and value == "":
        return b""
    raw = value.encode("utf-8")
    return encode_tag(field_number, WIRE_BYTES) + encode_varint(len(raw)) + raw


def f_msg(field_number: int, encoded_msg: bytes | None, *, omit_empty: bool = True) -> bytes:
    if encoded_msg is None:
        return b""
    if omit_empty and len(encoded_msg) == 0:
        return b""
    return encode_tag(field_number, WIRE_BYTES) + encode_varint(len(encoded_msg)) + encoded_msg


def f_double(field_number: int, value: float | None, *, omit_zero: bool = True) -> bytes:
    if value is None:
        return b""
    if omit_zero and value == 0.0:
        return b""
    return encode_tag(field_number, WIRE_FIXED64) + struct.pack("<d", value)


def f_float(field_number: int, value: float | None, *, omit_zero: bool = True) -> bytes:
    if value is None:
        return b""
    if omit_zero and value == 0.0:
        return b""
    return encode_tag(field_number, WIRE_FIXED32) + struct.pack("<f", value)


def join_fields(*parts: bytes) -> bytes:
    return b"".join(parts)


def encode_timestamp_ms(ms: int | None) -> bytes:
    """Encode Unix epoch milliseconds as google.protobuf.Timestamp."""
    if not ms or ms <= 0:
        return b""
    seconds = ms // 1000
    nanos = (ms % 1000) * 1_000_000
    return join_fields(
        f_varint(1, seconds),
        f_varint(2, nanos),
    )


def encode_timestamp_sec_nanos(seconds: int, nanos: int = 0) -> bytes:
    if seconds <= 0 and nanos <= 0:
        return b""
    return join_fields(
        f_varint(1, seconds),
        f_varint(2, nanos),
    )


def encode_duration_ms(ms: int | None) -> bytes:
    """Encode milliseconds as google.protobuf.Duration."""
    if not ms or ms <= 0:
        return b""
    seconds = ms // 1000
    nanos = (ms % 1000) * 1_000_000
    return join_fields(
        f_varint(1, seconds),
        f_varint(2, nanos),
    )


def encode_any(type_url: str, value: bytes) -> bytes:
    """Encode google.protobuf.Any { string type_url = 1; bytes value = 2; }."""
    return join_fields(
        f_str(1, type_url),
        f_bytes(2, value, omit_empty=False),
    )


def encode_str_map_entries(field_number: int, items: Iterable[Tuple[str, str]]) -> bytes:
    """Encode repeated map<string, string> entries { string key = 1; string value = 2; }."""
    out = bytearray()
    for k, v in items:
        entry = join_fields(
            f_str(1, k, omit_empty=False),
            f_str(2, v, omit_empty=False),
        )
        out.extend(f_msg(field_number, entry, omit_empty=False))
    return bytes(out)


def decode_fields(buf: bytes) -> List[Tuple[int, int, int | bytes]]:
    """Decode raw protobuf bytes into list of (field_number, wire_type, value)."""
    out: List[Tuple[int, int, int | bytes]] = []
    i = 0
    n = len(buf)
    while i < n:
        tag = 0
        shift = 0
        while i < n:
            b = buf[i]
            i += 1
            tag |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
            if shift > 63:
                raise ValueError(f"varint tag overflow at byte {i}")
        field_num = tag >> 3
        wire_type = tag & 7
        if wire_type == WIRE_VARINT:
            val = 0
            shift = 0
            while i < n:
                b = buf[i]
                i += 1
                val |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
                if shift > 63:
                    raise ValueError(f"varint value overflow at byte {i}")
            out.append((field_num, wire_type, val))
        elif wire_type == WIRE_FIXED64:
            if i + 8 > n:
                raise ValueError("truncated fixed64")
            out.append((field_num, wire_type, buf[i : i + 8]))
            i += 8
        elif wire_type == WIRE_BYTES:
            ln = 0
            shift = 0
            while i < n:
                b = buf[i]
                i += 1
                ln |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
                if shift > 63:
                    raise ValueError("varint length overflow")
            if i + ln > n:
                raise ValueError(f"truncated length-delimited field {field_num}: need {ln}, have {n - i}")
            out.append((field_num, wire_type, buf[i : i + ln]))
            i += ln
        elif wire_type == WIRE_FIXED32:
            if i + 4 > n:
                raise ValueError("truncated fixed32")
            out.append((field_num, wire_type, buf[i : i + 4]))
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wire_type} at byte {i}")
    return out
