"""Abstract base for audio filters applied by the transport layer."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class BaseAudioFilter(ABC):
    """Filter applied to incoming audio chunks before they leave the transport.

    Subclasses implement ``filter()`` to perform signal processing
    (denoising, gating, etc.) on raw audio bytes. The base ``process()``
    wrapper provides exception safety and automatic disabling of
    misbehaving filters.
    """

    enabled: bool = True

    async def process(self, data: bytes) -> Optional[bytes]:
        """Run the filter with exception safety. Returns *None* to drop the chunk."""
        if not self.enabled or not data:
            return data

        try:
            return await self.filter(data)
        except Exception:
            logger.error(
                "[%s] filter() raised; disabling denoising for this session",
                type(self).__name__,
                exc_info=True,
            )
            self.enabled = False
            return data

    @abstractmethod
    async def filter(self, data: bytes) -> Optional[bytes]:
        """Process a single audio chunk. Return *None* to drop it."""
        ...

    async def cleanup(self) -> None:
        """Called when the transport stops. Override for teardown."""
