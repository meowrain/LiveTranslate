"""DashScope Qwen-ASR-Realtime engine — cloud-based streaming ASR via WebSocket.

Supports two modes:
- Manual mode: persistent connection, append_audio + commit per segment
- Server VAD mode: per-segment connection, server detects speech boundaries
"""

import base64
import logging
import threading

import numpy as np

log = logging.getLogger("LiveTranslate.DashScope")

# Available models
DASHSCOPE_MODELS = {
    "qwen3-asr-flash-realtime": "Qwen3 ASR Flash (Realtime)",
}


class DashScopeASREngine:
    """Speech-to-text using DashScope Qwen-ASR-Realtime (cloud WebSocket)."""

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-asr-flash-realtime",
        language: str | None = None,
        server_vad: bool = False,
        vad_silence_ms: int = 400,
    ):
        if not api_key:
            raise ValueError("DashScope API key is required")
        try:
            from dashscope.audio.qwen_omni import OmniRealtimeConversation  # noqa: F401
        except ImportError:
            raise ImportError(
                "dashscope SDK not installed. Run: pip install dashscope>=1.25.6"
            )
        self._api_key = api_key
        self._model = model
        self.language = language
        self._server_vad = server_vad
        self._vad_silence_ms = vad_silence_ms
        self._conversation = None
        self._callback = None
        self._connected = False
        self._lock = threading.Lock()
        log.info(
            f"DashScope ASR created: model={model}, server_vad={server_vad}, "
            f"lang={language}, silence_ms={vad_silence_ms}"
        )

    def _connect(self) -> tuple:
        """Create a new connection and configure session. Returns (conversation, callback)."""
        from dashscope.audio.qwen_omni import (
            OmniRealtimeConversation,
            MultiModality,
        )
        from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams

        cb = _DashScopeCallback()
        conv = OmniRealtimeConversation(
            model=self._model,
            url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
            api_key=self._api_key,
            callback=cb,
        )
        cb._conversation = conv
        conv.connect()

        lang = self.language if self.language and self.language != "auto" else None
        transcription_params = TranscriptionParams(
            language=lang,
            sample_rate=16000,
            input_audio_format="pcm",
        )

        if self._server_vad:
            conv.update_session(
                output_modalities=[MultiModality.TEXT],
                enable_input_audio_transcription=True,
                transcription_params=transcription_params,
                enable_turn_detection=True,
                turn_detection_type="server_vad",
                turn_detection_threshold=0.0,
                turn_detection_silence_duration_ms=self._vad_silence_ms,
            )
        else:
            conv.update_session(
                output_modalities=[MultiModality.TEXT],
                enable_input_audio_transcription=True,
                transcription_params=transcription_params,
                enable_turn_detection=False,
            )

        return conv, cb

    def _ensure_connected(self) -> tuple:
        """Return (conversation, callback), reconnecting if needed.

        For manual mode: reuse persistent connection.
        For server VAD mode: create fresh connection each time (end_session kills it).
        """
        # Server VAD always needs a fresh connection
        if self._server_vad:
            conv, cb = self._connect()
            self._conversation = conv
            self._callback = cb
            self._connected = True
            return conv, cb

        # Manual mode: reuse existing connection
        if self._connected and self._conversation is not None and self._callback is not None:
            return self._conversation, self._callback

        try:
            conv, cb = self._connect()
            self._conversation = conv
            self._callback = cb
            self._connected = True
            log.info(f"DashScope connected: model={self._model}")
            return conv, cb
        except Exception as e:
            log.error(f"DashScope connect failed: {e}", exc_info=True)
            self._connected = False
            self._conversation = None
            self._callback = None
            raise

    def _close(self):
        """Close the current connection."""
        if self._conversation is not None:
            try:
                self._conversation.close()
            except Exception:
                pass
        self._conversation = None
        self._callback = None
        self._connected = False

    def _send_audio(self, conv, audio_b64: str):
        """Send base64 audio data in chunks."""
        chunk_size = 32 * 1024  # 32KB chunks
        for i in range(0, len(audio_b64), chunk_size):
            chunk = audio_b64[i : i + chunk_size]
            conv.append_audio(chunk)

    def transcribe(self, audio: np.ndarray) -> dict | None:
        """Transcribe an audio segment.

        Args:
            audio: float32 numpy array, 16kHz mono

        Returns:
            dict with 'text', 'language', 'language_name' or None.
        """
        with self._lock:
            try:
                conv, cb = self._ensure_connected()
            except Exception:
                return None

            if conv is None or cb is None:
                return None

            cb._result = None
            cb._result_ready.clear()
            cb._session_finished.clear()
            cb._error = None

            try:
                pcm_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                pcm_bytes = pcm_int16.tobytes()
                audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")

                self._send_audio(conv, audio_b64)

                if self._server_vad:
                    # Server VAD: end session to trigger server-side speech detection
                    conv.end_session(timeout=30)
                else:
                    # Manual mode: commit triggers recognition, session stays alive
                    conv.commit()

                timeout = max(len(audio) / 16000 + 5.0, 10.0)

                if cb._result_ready.wait(timeout=timeout):
                    result = cb._result
                    # Server VAD: connection is dead after end_session
                    if self._server_vad:
                        self._close()
                    if result:
                        log.debug(f"DashScope result: {result['text'][:80]}")
                        return result
                    if cb._error:
                        log.warning(f"DashScope error: {cb._error}")
                    return None
                else:
                    log.warning(
                        f"DashScope timeout ({timeout:.1f}s) waiting for result"
                    )
                    self._close()
                    return None
            except Exception as e:
                log.error(f"DashScope transcribe error: {e}", exc_info=True)
                self._close()
                return None

    def set_language(self, language: str):
        old = self.language
        self.language = language if language != "auto" else None
        log.info(f"DashScope language: {old} -> {self.language}")
        # Language change requires reconnection
        if self._connected:
            self._close()

    def unload(self):
        with self._lock:
            self._close()
        log.info("DashScope ASR unloaded")

    def to_device(self, device: str) -> bool:
        return True


# DashScope language code -> our standard code
_LANG_MAP = {
    "zh": "zh",
    "yue": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "de": "de",
    "ru": "ru",
    "fr": "fr",
    "pt": "pt",
    "ar": "ar",
    "it": "it",
    "es": "es",
    "hi": "hi",
    "id": "id",
    "th": "th",
    "tr": "tr",
    "uk": "uk",
    "vi": "vi",
}


class _DashScopeCallback:
    """Callback handler for DashScope real-time ASR events."""

    def __init__(self):
        self._conversation = None
        self._result: dict | None = None
        self._result_ready = threading.Event()
        self._session_finished = threading.Event()
        self._error: str | None = None

    def on_open(self):
        log.debug("DashScope WebSocket opened")

    def on_close(self, close_status_code, close_msg):
        log.debug(
            f"DashScope WebSocket closed: code={close_status_code}, msg={close_msg}"
        )
        self._session_finished.set()

    def on_event(self, message: dict):
        try:
            event_type = message.get("type", "")

            if event_type == "error":
                error = message.get("error", {})
                self._error = error.get("message", str(error))
                log.error(f"DashScope server error: {self._error}")
                self._result_ready.set()

            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = message.get("transcript", "").strip()
                if transcript:
                    self._result = {
                        "text": transcript,
                        "language": self._guess_lang(transcript),
                        "language_name": self._guess_lang(transcript),
                    }
                self._result_ready.set()

            elif event_type == "input_audio_buffer.speech_started":
                log.debug("DashScope: speech started")

            elif event_type == "input_audio_buffer.speech_stopped":
                log.debug("DashScope: speech stopped")

            elif event_type == "session.finished":
                log.debug("DashScope: session finished")
                self._session_finished.set()
                if not self._result_ready.is_set():
                    self._result_ready.set()

        except Exception as e:
            log.error(f"DashScope callback error: {e}", exc_info=True)
            self._error = str(e)
            self._result_ready.set()

    @staticmethod
    def _guess_lang(text: str) -> str:
        """Heuristic language detection from text content."""
        for ch in text:
            cp = ord(ch)
            if 0x4E00 <= cp <= 0x9FFF:
                return "zh"
            if 0x3040 <= cp <= 0x30FF:
                return "ja"
            if 0xAC00 <= cp <= 0xD7AF:
                return "ko"
        return "en"
