"""Abstract base for audio input filters applied by the transport layer."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class AudioInputFilter(ABC):
    """Filter applied to incoming audio chunks before they leave the transport.

    Subclasses implement ``filter()`` to perform signal processing
    (denoising, gating, etc.) on raw audio bytes. The base ``process()``
    wrapper provides exception safety and automatic disabling of
    misbehaving filters.
    """

    _disabled: bool = False

    async def process(self, data: bytes) -> Optional[bytes]:
        """Run the filter with exception safety. Returns *None* to drop the chunk."""
        if self._disabled:
            return data

        try:
            return await self.filter(data)
        except Exception as exc:
            logger.warning(
                "[AudioInputFilter] %s raised %s: %s — disabling for this session.",
                type(self).__name__, type(exc).__name__, exc,
            )
            self._disabled = True
            return data

    @abstractmethod
    async def filter(self, data: bytes) -> Optional[bytes]:
        """Process a single audio chunk. Return *None* to drop it."""
        ...

    async def setup(self) -> None:
        """Called when the transport starts. Override for initialisation."""

    async def cleanup(self) -> None:
        """Called when the transport stops. Override for teardown."""
