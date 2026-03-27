"""Centralized logging and telemetry setup for the Gemini Live Framework.

Usage (call once at application startup, before any other imports that log):

    from framework.logger import setup_logging, setup_telemetry

    setup_logging()       # configures native Python logging
    setup_telemetry()     # activates gemini-live-telemetry (if enabled via env)

Environment variables:

    LOG_LEVEL        — DEBUG, INFO, WARNING, ERROR, DISABLED  (default: INFO)
    TELEMETRY_MODE   — disabled, local, cloud                 (default: disabled)
"""

import logging
import os
import sys

# ---------------------------------------------------------------------------
# ANSI color codes
# ---------------------------------------------------------------------------
RESET = "\033[0m"
DIM = "\033[2m"
MAGENTA = "\033[95m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD_RED = "\033[1;31m"

LEVEL_COLORS = {
    logging.DEBUG: DIM,
    logging.INFO: GREEN,
    logging.WARNING: YELLOW,
    logging.ERROR: RED,
    logging.CRITICAL: BOLD_RED,
}


class ColorFormatter(logging.Formatter):
    """Colored log formatter: cyan timestamp | colored level | message."""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, self.datefmt)
        level = record.levelname.ljust(5)
        level_color = LEVEL_COLORS.get(record.levelno, RESET)
        msg = record.getMessage()

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            msg = msg + "\n" + record.exc_text
        if record.stack_info:
            msg = msg + "\n" + self.formatStackInfo(record.stack_info)

        return (
            f"{CYAN}{ts}{RESET} | "
            f"{level_color}{level}{RESET} | "
            f"{msg}"
        )


PLAIN_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger with colored (TTY) or plain (non-TTY) output.

    Args:
        level: Override log level.  Falls back to the ``LOG_LEVEL`` env var,
               then to ``"INFO"``.  Set to ``"DISABLED"`` to silence all logs.
    """
    level_str = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    if level_str == "DISABLED":
        logging.disable(logging.CRITICAL)
        return

    log_level = getattr(logging, level_str, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)

    if sys.stderr.isatty():
        formatter = ColorFormatter(datefmt=DATE_FORMAT)
    else:
        formatter = logging.Formatter(PLAIN_FORMAT, datefmt=DATE_FORMAT)

    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)
    root.addHandler(handler)


def setup_telemetry() -> None:
    """Activate gemini-live-telemetry based on the ``TELEMETRY_MODE`` env var.

    Modes:
        disabled  — no telemetry (default, no import cost)
        local     — JSON metrics to ./metrics/, no GCP export
        cloud     — full Cloud Monitoring export + auto-created dashboard + local JSON
    """
    mode = os.getenv("TELEMETRY_MODE", "disabled").lower()

    if mode == "disabled":
        return

    from gemini_live_telemetry import activate, InstrumentationConfig

    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

    if mode == "local":
        activate(InstrumentationConfig(
            project_id=project_id,
            enable_gcp_export=False,
            enable_dashboard=False,
            enable_json_export=True,
        ))
    elif mode == "cloud":
        activate(InstrumentationConfig(
            project_id=project_id,
            enable_gcp_export=True,
            enable_dashboard=True,
            enable_json_export=True,
        ))

    logger = logging.getLogger(__name__)
    logger.info(f"[Logger] Telemetry activated | mode={mode}")
