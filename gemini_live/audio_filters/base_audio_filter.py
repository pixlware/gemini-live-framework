"""Abstract base for audio filters applied by the transport layer."""

from abc import ABC, abstractmethod
from typing import Optional

from ..logger import SessionLogger

class BaseAudioFilter(ABC):
    """Filter applied to incoming audio chunks before they leave the transport.

    Subclasses implement ``filter()`` to perform signal processing
    (denoising, gating, etc.) on raw audio bytes. The base ``process()``
    wrapper provides exception safety and automatic disabling of
    misbehaving filters.
    """

    enabled: bool = True
    _session_logger: Optional[SessionLogger] = None

    @property
    def logger(self) -> SessionLogger:
        if self._session_logger is None:
            self._session_logger = SessionLogger()
        return self._session_logger

    @logger.setter
    def logger(self, logger: SessionLogger) -> None:
        self._session_logger = logger

    async def process(self, data: bytes) -> Optional[bytes]:
        """Run the filter with exception safety. Returns *None* to drop the chunk."""
        if not self.enabled or not data:
            return data

        try:
            return await self.filter(data)
        except Exception as exc:
            self.logger.error(
                f"[{type(self).__name__}] filter() raised; disabling denoising for this session",
                error=str(exc)
            )
            self.enabled = False
            return data

    @abstractmethod
    async def filter(self, data: bytes) -> Optional[bytes]:
        """Process a single audio chunk. Return *None* to drop it."""
        ...

    async def cleanup(self) -> None:
        """Called when the transport stops. Override for teardown."""
