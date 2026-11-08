"""Small, non-executing reader for TFRecord files containing tf.train.Example.

This avoids installing TensorFlow solely to read the official ASIMOV release.
Only protobuf wire types and Example's three feature-list types are supported.
"""

from __future__ import annotations

import pathlib
import struct


def _varint(data: bytes, position: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            raise ValueError("Truncated protobuf varint")
        byte = data[position]
        position += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, position
    raise ValueError("Oversized protobuf varint")


def _fields(data: bytes):
    position = 0
    while position < len(data):
        tag, position = _varint(data, position)
        field, wire = tag >> 3, tag & 7
        if field == 0:
            raise ValueError("Invalid protobuf field number")
        if wire == 0:
            value, position = _varint(data, position)
        elif wire in (1, 5):
            width = 8 if wire == 1 else 4
            value = data[position : position + width]
            position += width
            if len(value) != width:
                raise ValueError("Truncated fixed-width protobuf value")
        elif wire == 2:
            size, position = _varint(data, position)
            value = data[position : position + size]
            position += size
            if len(value) != size:
                raise ValueError("Truncated protobuf byte string")
        else:
            raise ValueError(f"Unsupported protobuf wire type: {wire}")
        yield field, wire, value


def decode_example(data: bytes) -> dict[str, list]:
    examples = [value for field, wire, value in _fields(data) if field == 1 and wire == 2]
    if len(examples) != 1:
        raise ValueError("Expected one Example.features message")
    result = {}
    for field, wire, entry in _fields(examples[0]):
        if field != 1 or wire != 2:
            raise ValueError("Unexpected Features entry")
        pairs = list(_fields(entry))
        key = next(value.decode("utf-8") for number, kind, value in pairs if number == 1 and kind == 2)
        feature = next(value for number, kind, value in pairs if number == 2 and kind == 2)
        feature_fields = list(_fields(feature))
        if len(feature_fields) != 1:
            raise ValueError("Expected one Feature list type")
        list_type, kind, values = feature_fields[0]
        if kind != 2:
            raise ValueError("Invalid Feature list encoding")
        decoded = []
        for number, encoding, value in _fields(values):
            if number != 1:
                raise ValueError("Unexpected Feature list field")
            if list_type == 1 and encoding == 2:
                decoded.append(value)
            elif list_type == 3 and encoding == 0:
                decoded.append(value if value < 2**63 else value - 2**64)
            elif list_type == 3 and encoding == 2:
                cursor = 0
                while cursor < len(value):
                    item, cursor = _varint(value, cursor)
                    decoded.append(item if item < 2**63 else item - 2**64)
            elif list_type == 2 and encoding == 2:
                if len(value) % 4:
                    raise ValueError("Malformed packed FloatList")
                decoded.extend(struct.unpack("<" + "f" * (len(value) // 4), value))
            else:
                raise ValueError("Unsupported Feature list encoding")
        if key in result:
            raise ValueError(f"Duplicate feature: {key}")
        result[key] = decoded
    return result


def _crc_table() -> tuple[int, ...]:
    table = []
    for value in range(256):
        crc = value
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
        table.append(crc)
    return tuple(table)


CRC_TABLE = _crc_table()


def _masked_crc32c(data: bytes) -> int:
    value = 0xFFFFFFFF
    for byte in data:
        value = CRC_TABLE[(value ^ byte) & 255] ^ (value >> 8)
    value ^= 0xFFFFFFFF
    return (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def iter_examples(path: pathlib.Path):
    with path.open("rb") as stream:
        while True:
            length_bytes = stream.read(8)
            if not length_bytes:
                return
            length_crc = stream.read(4)
            if len(length_bytes) != 8 or len(length_crc) != 4:
                raise ValueError("Truncated TFRecord header")
            length = struct.unpack("<Q", length_bytes)[0]
            if length > 64 * 1024 * 1024:
                raise ValueError("Unexpectedly large TFRecord example")
            if _masked_crc32c(length_bytes) != struct.unpack("<I", length_crc)[0]:
                raise ValueError("TFRecord length CRC mismatch")
            record, record_crc = stream.read(length), stream.read(4)
            if len(record) != length or len(record_crc) != 4:
                raise ValueError("Truncated TFRecord data")
            if _masked_crc32c(record) != struct.unpack("<I", record_crc)[0]:
                raise ValueError("TFRecord data CRC mismatch")
            yield decode_example(record)
