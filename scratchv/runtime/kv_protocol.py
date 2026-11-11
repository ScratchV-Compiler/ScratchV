"""Bounded little-endian stream framing for the resident KV guest (v1)."""
from __future__ import annotations
from dataclasses import dataclass
import struct

MAGIC = b'SVKV0001'
VERSION = 1
CAPACITY = 256
HEADER = struct.Struct('<8sIIIIQQIIII')
MAX_PAYLOAD = CAPACITY * 8
RESET, PREFILL, DECODE, CLOSE = 1, 2, 3, 4
READY, ACK, DONE, ERROR = 10, 11, 12, 13
KINDS = {RESET, PREFILL, DECODE, CLOSE, READY, ACK, DONE, ERROR}


def checksum(data):
    value = 2166136261
    for byte in data:
        value = ((value ^ byte) * 16777619) & 0xffffffff
    return value


@dataclass(frozen=True)
class Frame:
    kind: int
    session: int
    request: int
    valid_length: int = 0
    logits_bytes: int = 0
    status: int = 0
    payload: bytes = b''

    def encode(self):
        if self.kind not in KINDS or len(self.payload) > MAX_PAYLOAD:
            raise ValueError('Invalid frame kind/payload length')
        header = HEADER.pack(MAGIC, VERSION, self.kind, len(self.payload), self.status,
                             self.session, self.request, self.valid_length, self.logits_bytes, 0, CAPACITY)
        digest = checksum(header + self.payload)
        return header[:48] + struct.pack('<I', digest) + header[52:] + self.payload


class FrameDecoder:
    """Incremental stream decoder. Invalid bytes poison the caller's session."""
    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data):
        self.buffer.extend(data)
        frames = []
        while len(self.buffer) >= HEADER.size:
            magic, version, kind, length, status, session, request, valid, size, digest, capacity = HEADER.unpack_from(self.buffer)
            if (magic != MAGIC or version != VERSION or kind not in KINDS
                    or length > MAX_PAYLOAD or capacity != CAPACITY or valid > CAPACITY):
                raise ValueError('Invalid KV protocol header')
            end = HEADER.size + length
            if len(self.buffer) < end:
                break
            packet = bytes(self.buffer[:end])
            del self.buffer[:end]
            if checksum(packet[:48] + b'\0'*4 + packet[52:]) != digest:
                raise ValueError('Invalid KV frame checksum')
            frames.append(Frame(kind, session, request, valid, size, status, packet[HEADER.size:]))
        return frames

    def eof(self):
        if self.buffer:
            raise EOFError('Truncated KV protocol frame')
        raise EOFError('KV protocol connection closed')
