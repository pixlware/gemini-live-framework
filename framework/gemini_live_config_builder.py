"""Factory for building a types.LiveConnectConfig with sensible defaults.

Only convenience shortcuts for deeply nested SDK structures live here.
All other LiveConnectConfig fields pass straight through via **kwargs,
so the builder never needs updating when Google adds new fields.

When a consumer passes a partial SDK struct (e.g. ``speech_config``
with only ``language_code`` set), the explicitly-set fields are merged
on top of the defaults rather than replacing them wholesale.
"""

import logging
from typing import Optional

import pydantic
from google.genai import types

logger = logging.getLogger(__name__)


def build_gemini_live_config(
    *,
    # ── Convenience shortcuts (expand to nested SDK structures) ──
    voice_name: str = "Zephyr",
    language_code: Optional[str] = None,
    function_declarations: Optional[list[types.FunctionDeclaration]] = None,
    rag_corpus: Optional[str] = None,
    vad_enabled: bool = True,
    # ── Everything else forwards to LiveConnectConfig as-is ──
    **kwargs,
) -> types.LiveConnectConfig:
    """Build a LiveConnectConfig with battle-tested defaults.

    Convenience shortcuts handle the deeply nested structures (speech,
    VAD, tools, transcription) so callers don't have to.  Any kwarg that
    matches a LiveConnectConfig field is **shallow-merged** on top of the
    corresponding default — only the fields the caller explicitly set
    replace the defaults; the rest are kept.

    Returns a ``types.LiveConnectConfig`` that callers can further
    mutate before handing to ``GeminiLiveSession``.
    """

    # ── speech_config ──
    default_speech = types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name=voice_name,
            )
        ),
        language_code=language_code,
    )
    kwargs["speech_config"] = _merge_with_default(default_speech, kwargs.get("speech_config"))

    # ── tools ──
    if "tools" not in kwargs:
        tools = _build_tools(function_declarations, rag_corpus)
        if tools:
            kwargs["tools"] = tools

    # ── transcription defaults ──
    default_transcription = types.AudioTranscriptionConfig(language_codes=["en-US"])

    kwargs["input_audio_transcription"] = _merge_with_default(
        default_transcription, kwargs.get("input_audio_transcription"),
    )
    kwargs["output_audio_transcription"] = _merge_with_default(
        default_transcription, kwargs.get("output_audio_transcription"),
    )

    # ── VAD / realtime input ──
    if vad_enabled:
        default_vad = types.AutomaticActivityDetection(
            disabled=False,
            start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_UNSPECIFIED,
            end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
            prefix_padding_ms=300,
            silence_duration_ms=800,
        )
    else:
        default_vad = types.AutomaticActivityDetection(disabled=True)

    default_realtime = types.RealtimeInputConfig(
        automatic_activity_detection=default_vad,
    )
    kwargs["realtime_input_config"] = _merge_with_default(
        default_realtime, kwargs.get("realtime_input_config"),
    )

    if "explicit_vad_signal" not in kwargs:
        kwargs["explicit_vad_signal"] = vad_enabled

    # ── response modalities ──
    if "response_modalities" not in kwargs:
        kwargs["response_modalities"] = ["AUDIO"]

    return types.LiveConnectConfig(**kwargs)


# --- Helpers ---------------------------------------------------------------

def _merge_with_default(default: pydantic.BaseModel, override: Optional[pydantic.BaseModel]):
    """Shallow-merge *override*'s explicitly-set fields on top of *default*.

    Uses Pydantic v2's ``model_fields_set`` to know which fields the
    caller actually provided vs. which are just Pydantic defaults.
    Returns *default* untouched when *override* is ``None``.
    """
    if override is None:
        return default
    explicit = {f: getattr(override, f) for f in override.model_fields_set}
    if not explicit:
        return default
    return default.model_copy(update=explicit)


def _build_tools(
    function_declarations: Optional[list[types.FunctionDeclaration]] = None,
    rag_corpus: Optional[str] = None,
) -> Optional[list[types.Tool]]:
    """Assemble a tools list from function declarations and/or a RAG corpus."""
    tools: list[types.Tool] = []

    if rag_corpus:
        tools.append(
            types.Tool(
                retrieval=types.Retrieval(
                    vertex_rag_store=types.VertexRagStore(
                        rag_resources=[
                            types.VertexRagStoreRagResource(rag_corpus=rag_corpus)
                        ]
                    )
                )
            )
        )

    if function_declarations:
        tools.append(types.Tool(function_declarations=function_declarations))

    return tools or None
