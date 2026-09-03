"""PulseVAD: ultra-tiny 2.1k streaming VAD for microcontrollers and edge devices."""

from pulsevad.model import PulseVAD
from pulsevad.frontend import MelFrontend
from pulsevad.utils_vad import (
    load_pulsevad,
    read_audio,
    predict_window,
    get_speech_timestamps,
    get_model_path,
)

__version__ = "0.1.0"

__all__ = [
    "PulseVAD",
    "MelFrontend",
    "load_pulsevad",
    "read_audio",
    "predict_window",
    "get_speech_timestamps",
    "get_model_path",
    "__version__",
]
