"""Accumulates audio chunks into fixed-size windows."""

from typing import Iterator


class BufferService:
    """Buffers incoming audio and yields complete chunks of ``chunk_size`` bytes."""

    chunk_size: int
    _buffer: bytearray

    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
        self._buffer = bytearray()

    def get_chunks(self, audio_data: bytes) -> Iterator[bytes]:
        """Append audio_data and yield all complete chunks."""
        if not audio_data:
            return
        self._buffer.extend(audio_data)
        while len(self._buffer) >= self.chunk_size:
            yield bytes(self._buffer[:self.chunk_size])
            del self._buffer[:self.chunk_size]

    def flush(self) -> bytes:
        """Return the remaining buffer padded with zeros to chunk_size, and clear it."""
        if not self._buffer:
            return b""
        remainder = len(self._buffer)
        padding_size = self.chunk_size - remainder
        padded_chunk = bytes(self._buffer) + (b"\x00" * padding_size)
        self._buffer.clear()
        return padded_chunk
