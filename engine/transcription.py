# engine/transcription.py
from __future__ import annotations
import json
import os
import logging

# 1. КРИТИЧНО ДЛЯ WINDOWS: предотвращает внезапное закрытие процесса (Crash)
# из-за конфликта OpenMP (libiomp5md.dll) между NumPy/SciPy, PyQt6 и CTranslate2 (faster-whisper)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Подавляем лишние предупреждения от huggingface_hub в консоли
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("faster_whisper").setLevel(logging.ERROR)

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
import numpy as np


@dataclass
class TranscriptWord:
    word_id: str
    text: str
    start_sec: float
    end_sec: float
    confidence: float = 0.99

    def start_frame(self, sample_rate: int) -> int:
        return max(0, int(round(self.start_sec * sample_rate)))

    def end_frame(self, sample_rate: int) -> int:
        return max(0, int(round(self.end_sec * sample_rate)))


class TranscriptionEngine:
    """
    Стабильный движок распознавания речи faster-whisper для Windows / macOS / Linux.
    Безопасно скачивает модель в локальную папку без симлинков и конфликтов OpenMP.
    """

    def __init__(self, model_size: str = "base", language: str = "ru",
                 models_dir: Path | None = None) -> None:
        self.model_size = model_size
        self.language = language
        self.models_dir = models_dir or (Path.home() / ".ltstudio_workspace" / "models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._whisper_model = None
        self.last_error: str = ""

    def _ensure_model_downloaded(self) -> Path:
        """Скачивает файлы модели напрямую в папку ~/.ltstudio_workspace/models/whisper_<size>."""
        from huggingface_hub import snapshot_download

        target_dir = self.models_dir / f"whisper_{self.model_size}"
        model_bin = target_dir / "model.bin"
        if not model_bin.exists():
            repo_id = f"Systran/faster-whisper-{self.model_size}"
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )
        return target_dir

    def transcribe_audio(self, audio: np.ndarray, sample_rate: int) -> List[TranscriptWord]:
        if audio.size == 0:
            return []

        try:
            from faster_whisper import WhisperModel
            from scipy import signal

            if self._whisper_model is None:
                local_model_path = self._ensure_model_downloaded()
                self._whisper_model = WhisperModel(
                    str(local_model_path),
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=4,
                )

            mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio.copy()
            mono = mono.astype(np.float32)

            # Приводим к 16000 Гц (родная частота Whisper)
            target_sr = 16000
            if sample_rate != target_sr:
                target_len = max(1, int(round(len(mono) * target_sr / sample_rate)))
                mono = signal.resample(mono, target_len).astype(np.float32)

            # Нормализуем тихий сигнал перед подачей в нейросеть
            peak = float(np.max(np.abs(mono))) if mono.size > 0 else 0.0
            if peak > 1e-4:
                mono = mono / peak * 0.9

            # vad_filter=False исключает падение onnxruntime на Windows
            segments, _ = self._whisper_model.transcribe(
                mono,
                language=self.language,
                word_timestamps=True,
                vad_filter=False,
                condition_on_previous_text=False,
            )

            words: List[TranscriptWord] = []
            idx = 0
            for seg in segments:
                if seg.words:
                    for w in seg.words:
                        token = w.word.strip()
                        if not token:
                            continue
                        idx += 1
                        words.append(
                            TranscriptWord(
                                word_id=f"w_{idx}",
                                text=token,
                                start_sec=round(float(w.start), 3),
                                end_sec=round(float(w.end), 3),
                                confidence=round(float(w.probability), 3),
                            )
                        )
                else:
                    token = seg.text.strip()
                    if token:
                        idx += 1
                        words.append(
                            TranscriptWord(
                                word_id=f"w_{idx}",
                                text=token,
                                start_sec=round(float(seg.start), 3),
                                end_sec=round(float(seg.end), 3),
                                confidence=0.95,
                            )
                        )
            self.last_error = ""
            return words
        except Exception as exc:
            self.last_error = str(exc)
            return []

    @staticmethod
    def search_transcript(words: List[TranscriptWord], query: str) -> List[TranscriptWord]:
        q = query.strip().lower()
        if not q:
            return []
        return [w for w in words if q in w.text.lower()]

    @staticmethod
    def export_subtitles(words: List[TranscriptWord], fmt: str = "SRT",
                         words_per_cue: int = 7) -> str:
        fmt = fmt.upper()
        if fmt == "JSON":
            return json.dumps([asdict(w) for w in words], ensure_ascii=False, indent=2)
        if fmt == "TXT":
            return " ".join(w.text for w in words)

        cues = [words[i : i + words_per_cue] for i in range(0, len(words), words_per_cue)]
        lines: List[str] = []
        if fmt == "VTT":
            lines.append("WEBVTT\n")

        for idx, group in enumerate(cues, start=1):
            start_str = _format_timestamp(group[0].start_sec, vtt=(fmt == "VTT"))
            end_str = _format_timestamp(group[-1].end_sec, vtt=(fmt == "VTT"))
            text = " ".join(w.text for w in group)
            if fmt == "SRT":
                lines.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")
            else:
                lines.append(f"{start_str} --> {end_str}\n{text}\n")
        return "\n".join(lines)


def _format_timestamp(seconds: float, vtt: bool = False) -> str:
    ms = int(round((seconds % 1.0) * 1000))
    total_sec = int(seconds)
    s = total_sec % 60
    m = (total_sec // 60) % 60
    h = total_sec // 3600
    sep = "." if vtt else ","
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"