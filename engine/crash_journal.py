# engine/crash_journal.py
from __future__ import annotations
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import soundfile as sf


class CrashRecoveryJournal:
    """
    Атомарный журнал записи в реальном времени.
    Гарантирует сохранность записанного аудио при аварийном завершении процесса или ОС.
    """

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.session_id: str = uuid.uuid4().hex[:12]
        self.raw_path: Path = self.workspace_dir / f"rec_{self.session_id}.journal.raw"
        self.meta_path: Path = self.workspace_dir / f"rec_{self.session_id}.journal.json"
        self._raw_fd: Optional[int] = None
        self.frames_written: int = 0
        self.sample_rate: int = 48000
        self.channels: int = 1
        self.markers: List[Dict[str, Any]] = []
        self._blocks_since_sync: int = 0

    def start_session(self, sample_rate: int, channels: int, bit_depth: str = "FLOAT") -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames_written = 0
        self.markers.clear()
        self._raw_fd = os.open(str(self.raw_path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        self._write_atomic_metadata(status="RECORDING", bit_depth=bit_depth)

    def append_block(self, block: np.ndarray) -> None:
        """Запись блока float32 на диск с периодическим fsync."""
        if self._raw_fd is None:
            return
        contiguous = np.ascontiguousarray(block, dtype=np.float32)
        os.write(self._raw_fd, contiguous.tobytes())
        self.frames_written += contiguous.shape[0]
        self._blocks_since_sync += 1
        if self._blocks_since_sync >= 25:  # ~0.5 сек при буфере 1024 и 48 кГц
            os.fsync(self._raw_fd)
            self._write_atomic_metadata(status="RECORDING")
            self._blocks_since_sync = 0

    def add_live_marker(self, label: str) -> Dict[str, Any]:
        marker = {
            "id": uuid.uuid4().hex[:8],
            "label": label,
            "frame": self.frames_written,
            "time_sec": round(self.frames_written / max(1, self.sample_rate), 3),
        }
        self.markers.append(marker)
        self._write_atomic_metadata(status="RECORDING")
        return marker

    def finalize_to_audio_file(self, target_path: Path, subtype: str = "PCM_24") -> Path:
        """Штатное завершение записи: конвертация журнала в финальный файл и очистка журнала."""
        if self._raw_fd is not None:
            os.fsync(self._raw_fd)
            os.close(self._raw_fd)
            self._raw_fd = None

        audio = self._read_raw_buffer(self.raw_path, self.channels)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(target_path), audio, self.sample_rate, subtype=subtype)

        # Удаляем временные файлы журнала только после успешной записи целевого файла
        self.raw_path.unlink(missing_ok=True)
        self.meta_path.unlink(missing_ok=True)
        return target_path

    def _write_atomic_metadata(self, status: str, bit_depth: str = "PCM_24") -> None:
        tmp_path = self.meta_path.with_suffix(".tmp")
        payload = {
            "session_id": self.session_id,
            "status": status,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": bit_depth,
            "frames_written": self.frames_written,
            "markers": self.markers,
            "updated_at": time.time(),
            "raw_file": self.raw_path.name,
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.meta_path)

    @staticmethod
    def _read_raw_buffer(raw_path: Path, channels: int) -> np.ndarray:
        if not raw_path.exists() or raw_path.stat().st_size == 0:
            return np.zeros((0, channels), dtype=np.float32)
        raw_bytes = raw_path.read_bytes()
        usable_floats = (len(raw_bytes) // 4 // channels) * channels
        arr = np.frombuffer(raw_bytes[: usable_floats * 4], dtype=np.float32)
        return arr.reshape(-1, channels)

    @classmethod
    def recover_incomplete_sessions(cls, workspace_dir: Path) -> List[Path]:
        """Сканирует папку при запуске приложения и восстанавливает оборванные записи."""
        recovered_files: List[Path] = []
        workspace_dir = Path(workspace_dir)
        if not workspace_dir.exists():
            return recovered_files

        for meta_file in workspace_dir.glob("rec_*.journal.json"):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                raw_file = workspace_dir / meta["raw_file"]
                if meta.get("status") == "RECORDING" and raw_file.exists():
                    ch = int(meta.get("channels", 1))
                    sr = int(meta.get("sample_rate", 48000))
                    audio = cls._read_raw_buffer(raw_file, ch)
                    if audio.shape[0] > 0:
                        out_wav = workspace_dir / f"recovered_{meta['session_id']}.wav"
                        sf.write(str(out_wav), audio, sr, subtype="PCM_24")
                        recovered_files.append(out_wav)
                raw_file.unlink(missing_ok=True)
                meta_file.unlink(missing_ok=True)
            except Exception:
                continue
        return recovered_files