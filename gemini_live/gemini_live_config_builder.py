"""Factory for building a types.LiveConnectConfig with sensible defaults.

Only convenience shortcuts for deeply nested SDK structures live here.
All other LiveConnectConfig fields pass straight through via **kwargs,
so the builder never needs updating when Google adds new fields.

Sub-configs (``speech_config``, ``realtime_input_config``,
``input_audio_transcription``, ``output_audio_transcription``) replace
the framework defaults wholesale when passed. Use the shortcut kwargs
(``voice_name``, ``language_code``, ``vad_enabled``) OR a full sub-config,
not both — mixing them emits a warning and the sub-config wins.
"""

import logging
from typing import Optional

from google.genai import types

logger = logging.getLogger(__name__)


def build_gemini_live_config(
    *,
    # ── Convenience shortcuts (expand to nested SDK structures) ──
    voice_name: str = "Zephyr",
    language_code: Optional[str] = None,
    function_declarations: Optional[list[types.FunctionDeclaration]] = None,
    vad_enabled: bool = True,
    enable_google_search: bool = False,
    # ── Everything else forwards to LiveConnectConfig as-is ──
    **kwargs,
) -> types.LiveConnectConfig:
    """Build a LiveConnectConfig with battle-tested defaults.

    Convenience shortcuts (``voice_name``, ``language_code``,
    ``vad_enabled``) handle the deeply nested structures so callers
    don't have to. Any sub-config passed via ``**kwargs`` replaces the
    framework default wholesale — the shortcuts do not merge into it.
    Mixing a shortcut with its overlapping sub-config emits a warning;
    the sub-config wins.

    Returns a ``types.LiveConnectConfig`` that callers can further
    mutate before handing to ``GeminiLiveSession``.
    """

    # ── speech_config ──
    if "speech_config" in kwargs and (voice_name != "Zephyr" or language_code is not None):
        logger.warning(
            "[ConfigBuilder] speech_config= provided alongside voice_name/language_code "
            "shortcuts; speech_config wins and the shortcuts are ignored. "
            "Use one or the other."
        )
    default_speech = types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name=voice_name,
            )
        ),
        language_code=language_code,
    )
    kwargs.setdefault("speech_config", default_speech)

    # ── tools ──
    if "tools" not in kwargs:
        tools = _build_tools(
            function_declarations=function_declarations,
            enable_google_search=enable_google_search,
        )
        if tools:
            kwargs["tools"] = tools

    # ── transcription defaults ──
    default_transcription = types.AudioTranscriptionConfig(language_codes=["en-US"])
    kwargs.setdefault("input_audio_transcription", default_transcription)
    kwargs.setdefault("output_audio_transcription", default_transcription)

    # ── VAD / realtime input ──
    if "realtime_input_config" in kwargs and vad_enabled is not True:
        logger.warning(
            "[ConfigBuilder] realtime_input_config= provided alongside vad_enabled=False; "
            "realtime_input_config wins and vad_enabled is ignored."
        )
    if vad_enabled:
        default_vad = types.AutomaticActivityDetection(
            disabled=False,
            start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
            end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
            prefix_padding_ms=200,
            silence_duration_ms=600,
        )
    else:
        default_vad = types.AutomaticActivityDetection(disabled=True)

    default_realtime = types.RealtimeInputConfig(
        automatic_activity_detection=default_vad,
    )
    kwargs.setdefault("realtime_input_config", default_realtime)

    if "explicit_vad_signal" not in kwargs:
        kwargs["explicit_vad_signal"] = vad_enabled

    # ── response modalities ──
    if "response_modalities" not in kwargs:
        kwargs["response_modalities"] = ["AUDIO"]

    config = types.LiveConnectConfig(**kwargs)
    logger.debug(
        "[ConfigBuilder] Built LiveConnectConfig: %s",
        config.model_dump_json(exclude_none=True, exclude_defaults=True),
    )
    return config


# --- Helpers ---------------------------------------------------------------

def _build_tools(
    function_declarations: Optional[list[types.FunctionDeclaration]] = None,
    enable_google_search: bool = False,
) -> Optional[list[types.Tool]]:
    """Assemble a tools list from function declarations and/or a RAG corpus."""
    tools: list[types.Tool] = []

    if enable_google_search:
        tools.append(types.Tool(google_search=types.GoogleSearch()))

    if function_declarations:
        tools.append(types.Tool(function_declarations=function_declarations))

    return tools or None
