# engine/audio_engine.py
from __future__ import annotations
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import soundfile as sf

from engine.crash_journal import CrashRecoveryJournal
from engine.dsp_chain import DSPEffectChain, PitchTimeShiftPlugin
from project.edl import NonDestructiveTimeline


@dataclass
class AudioTelemetry:
    peak_db_l: float = -96.0
    peak_db_r: float = -96.0
    rms_db_l: float = -96.0
    rms_db_r: float = -96.0
    phase_correlation: float = 1.0
    current_frame: int = 0
    current_time_sec: float = 0.0
    is_clipping: bool = False
    feedback_warning: bool = False


class VoiceAudioEngine:
    """
    Независимый аудио-движок студийного класса LTStudio.
    Управляет записью (с выдачей живого буфера для отрисовки на дорожке),
    мониторингом, воспроизведением проекта и изолированным предпрослушиванием файлов.
    """

    SUPPORTED_SAMPLE_RATES = (44100, 48000, 96000)
    SUPPORTED_BIT_DEPTHS = ("PCM_16", "PCM_24", "FLOAT")

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.sample_rate: int = 48000
        self.channels: int = 1
        self.bit_depth: str = "PCM_24"
        self.block_size: int = 1024

        self.timeline = NonDestructiveTimeline(sample_rate=self.sample_rate, channels=self.channels)
        self.dsp_chain = DSPEffectChain()
        self.journal = CrashRecoveryJournal(self.workspace_dir / "journals")

        self.is_recording: bool = False
        self.is_paused: bool = False
        self.is_playing: bool = False
        self.is_previewing: bool = False
        self.monitoring_enabled: bool = False
        self.monitor_with_fx: bool = True
        self.playback_speed: float = 1.0
        self.loop_enabled: bool = False
        self.loop_range_frames: Optional[tuple[int, int]] = None

        self.telemetry = AudioTelemetry()
        self._record_queue: queue.SimpleQueue[Optional[np.ndarray]] = queue.SimpleQueue()
        self._disk_thread: Optional[threading.Thread] = None
        self._recorded_chunks: List[np.ndarray] = []
        self._play_buffer: np.ndarray = np.zeros((0, 1), dtype=np.float32)
        self._play_cursor: int = 0
        self._sd_stream: Any = None
        self._play_stream: Any = None

        self._preview_buffer: np.ndarray = np.zeros((0, 1), dtype=np.float32)
        self._preview_cursor: int = 0
        self._preview_stream: Any = None

    @staticmethod
    def list_audio_devices() -> List[Dict[str, Any]]:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            result = []
            for idx, d in enumerate(devices):
                result.append({
                    "id": idx,
                    "name": d["name"],
                    "max_input_channels": d["max_input_channels"],
                    "max_output_channels": d["max_output_channels"],
                    "default_samplerate": d["default_samplerate"],
                })
            return result
        except Exception:
            return [
                {"id": 0, "name": "System Default Microphone / Audio Interface", "max_input_channels": 2,
                 "max_output_channels": 2, "default_samplerate": 48000}
            ]

    def configure_audio(self, sample_rate: int = 48000, channels: int = 1,
                        bit_depth: str = "PCM_24", block_size: int = 1024) -> None:
        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Неподдерживаемая частота дискретизации: {sample_rate}")
        if bit_depth not in self.SUPPORTED_BIT_DEPTHS:
            raise ValueError(f"Неподдерживаемая разрядность: {bit_depth}")
        self.sample_rate = sample_rate
        self.channels = max(1, min(2, channels))
        self.bit_depth = bit_depth
        self.block_size = block_size
        self.timeline.sample_rate = self.sample_rate
        self.timeline.channels = self.channels
        self.dsp_chain.reset_all(self.sample_rate, self.channels)

    # ------------------------------------------------------------------
    # Запись, живой буфер для отрисовки и Crash-Recovery
    # ------------------------------------------------------------------

    def get_live_recording_mono(self) -> np.ndarray:
        """Возвращает текущий накопленный моно-массив во время записи для живой отрисовки на дорожке."""
        if not self.is_recording or not self._recorded_chunks:
            return np.zeros(0, dtype=np.float32)
        snapshot = list(self._recorded_chunks)
        if not snapshot:
            return np.zeros(0, dtype=np.float32)
        full = np.vstack(snapshot)
        return full[:, 0] if full.ndim > 1 else full

    def start_recording(self, input_device: Optional[int] = None) -> None:
        if self.is_recording:
            return
        self.stop_playback()
        self.stop_preview()

        self.is_recording = True
        self.is_paused = False
        self._recorded_chunks.clear()
        self.journal.start_session(self.sample_rate, self.channels, self.bit_depth)

        self._disk_thread = threading.Thread(target=self._disk_writer_worker, daemon=True)
        self._disk_thread.start()

        try:
            import sounddevice as sd
            if self.monitoring_enabled:
                self._sd_stream = sd.Stream(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    device=(input_device, None),
                    channels=self.channels,
                    dtype="float32",
                    callback=self._duplex_audio_callback,
                )
            else:
                self._sd_stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.block_size,
                    device=input_device,
                    channels=self.channels,
                    dtype="float32",
                    callback=self._input_audio_callback,
                )
            self._sd_stream.start()
        except Exception:
            self._sd_stream = None

    def push_input_block(self, block: np.ndarray) -> Optional[np.ndarray]:
        if not self.is_recording or self.is_paused:
            return np.zeros_like(block)

        if block.ndim == 1:
            block = block[:, np.newaxis]

        self._update_telemetry(block, frame_pos=self.journal.frames_written)
        self._record_queue.put(block.copy())
        self._recorded_chunks.append(block.copy())

        if self.monitoring_enabled:
            if self.telemetry.is_clipping:
                self.telemetry.feedback_warning = True
                return np.zeros_like(block)
            mon = self.dsp_chain.process(block, self.sample_rate) if self.monitor_with_fx else block
            return mon * 0.85
        return None

    def pause_recording(self) -> None:
        self.is_paused = True

    def resume_recording(self) -> None:
        self.is_paused = False

    def add_recording_marker(self, label: str) -> Dict[str, Any]:
        return self.journal.add_live_marker(label)

    def stop_recording(self, target_path: Optional[Path] = None) -> Path:
        if not self.is_recording:
            raise RuntimeError("Запись не запущена")
        self.is_recording = False
        self.is_paused = False

        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None

        self._record_queue.put(None)
        if self._disk_thread is not None:
            self._disk_thread.join(timeout=3.0)

        if target_path is None:
            target_path = self.workspace_dir / f"take_{int(time.time())}.wav"

        saved_path = self.journal.finalize_to_audio_file(target_path, subtype=self.bit_depth)
        if self._recorded_chunks:
            full_audio = np.vstack(self._recorded_chunks)
            self.timeline.register_source(
                saved_path.stem, full_audio, append_to_timeline=True, file_path=str(saved_path)
            )
        return saved_path

    def _disk_writer_worker(self) -> None:
        while True:
            block = self._record_queue.get()
            if block is None:
                break
            self.journal.append_block(block)

    def _input_audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        self.push_input_block(indata)

    def _duplex_audio_callback(self, indata: np.ndarray, outdata: np.ndarray,
                               frames: int, time_info: Any, status: Any) -> None:
        mon = self.push_input_block(indata)
        if mon is not None:
            outdata[:] = mon
        else:
            outdata.fill(0.0)

    # ------------------------------------------------------------------
    # Изолированное прослушивание файла из вкладки «Записи»
    # ------------------------------------------------------------------

    def start_file_preview(self, file_path: Path, output_device: Optional[int] = None) -> bool:
        if self.is_recording:
            return False
        self.stop_playback()
        self.stop_preview()

        try:
            audio, sr = sf.read(str(file_path), dtype="float32", always_2d=True)
            if audio.size == 0:
                return False
            self._preview_buffer = audio
            self._preview_cursor = 0
            self.is_previewing = True

            import sounddevice as sd
            self._preview_stream = sd.OutputStream(
                samplerate=sr,
                blocksize=self.block_size,
                device=output_device,
                channels=audio.shape[1],
                dtype="float32",
                callback=self._preview_audio_callback,
            )
            self._preview_stream.start()
            return True
        except Exception:
            self.is_previewing = False
            self._preview_stream = None
            return False

    def stop_preview(self) -> None:
        self.is_previewing = False
        if self._preview_stream is not None:
            try:
                self._preview_stream.stop()
                self._preview_stream.close()
            except Exception:
                pass
            self._preview_stream = None

    def _preview_audio_callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if not self.is_previewing or self._preview_buffer.size == 0:
            outdata.fill(0.0)
            import sounddevice as sd
            raise sd.CallbackStop()

        total = self._preview_buffer.shape[0]
        end_pos = self._preview_cursor + frames
        if end_pos <= total:
            chunk = self._preview_buffer[self._preview_cursor : end_pos]
            self._preview_cursor = end_pos
        else:
            part1 = self._preview_buffer[self._preview_cursor : total]
            rem = frames - len(part1)
            part2 = np.zeros((rem, self._preview_buffer.shape[1]), dtype=np.float32)
            chunk = np.vstack([part1, part2]) if len(part1) > 0 else part2
            self.is_previewing = False

        outdata[:] = chunk[:, : outdata.shape[1]]
        self._update_telemetry(chunk, frame_pos=self.telemetry.current_frame)

        if not self.is_previewing:
            import sounddevice as sd
            raise sd.CallbackStop()

    # ------------------------------------------------------------------
    # Воспроизведение многодорожечного проекта
    # ------------------------------------------------------------------

    def prepare_playback_buffer(self) -> np.ndarray:
        raw = self.timeline.render_timeline()
        self.dsp_chain.reset_all(self.sample_rate, self.timeline.channels)
        processed = self.dsp_chain.process(raw, self.sample_rate)
        if abs(self.playback_speed - 1.0) > 1e-2 and processed.size > 0:
            shifter = PitchTimeShiftPlugin(enabled=True)
            shifter.set_param("time_rate", float(np.clip(self.playback_speed, 0.5, 2.0)))
            processed = shifter.process_block(processed, self.sample_rate)
        if processed.ndim == 1:
            processed = processed[:, np.newaxis]
        self._play_buffer = processed
        return processed

    def seek(self, frame: int) -> None:
        total = self._play_buffer.shape[0] if self._play_buffer.size > 0 else self.timeline.total_frames
        self._play_cursor = max(0, min(max(0, total - 1), int(frame)))
        self.telemetry.current_frame = self._play_cursor
        self.telemetry.current_time_sec = self._play_cursor / max(1, self.sample_rate)

    def start_playback(self, start_frame: Optional[int] = None, output_device: Optional[int] = None) -> bool:
        if self.is_recording:
            return False
        self.stop_preview()
        self.stop_playback()
        self.prepare_playback_buffer()
        if self._play_buffer.shape[0] == 0:
            return False

        if start_frame is not None:
            if start_frame >= self._play_buffer.shape[0] - 64:
                start_frame = 0
            self.seek(start_frame)
        elif self._play_cursor >= self._play_buffer.shape[0] - 64:
            self._play_cursor = 0

        if self.loop_enabled and self.loop_range_frames is not None:
            ls, le = self.loop_range_frames
            if not (ls <= self._play_cursor < le):
                self._play_cursor = ls

        self.is_playing = True
        out_channels = self._play_buffer.shape[1]

        try:
            import sounddevice as sd
            self._play_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=output_device,
                channels=out_channels,
                dtype="float32",
                callback=self._output_audio_callback,
            )
            self._play_stream.start()
            return True
        except Exception:
            self._play_stream = None
            return True

    def stop_playback(self) -> None:
        self.is_playing = False
        if self._play_stream is not None:
            try:
                self._play_stream.stop()
                self._play_stream.close()
            except Exception:
                pass
            self._play_stream = None

    def _output_audio_callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if not self.is_playing:
            outdata.fill(0.0)
            import sounddevice as sd
            raise sd.CallbackStop()

        chunk = self.get_next_playback_block(frames)
        if chunk.ndim == 1:
            chunk = chunk[:, np.newaxis]

        if chunk.shape[1] == outdata.shape[1]:
            outdata[:] = chunk
        elif chunk.shape[1] == 1 and outdata.shape[1] > 1:
            outdata[:] = np.repeat(chunk, outdata.shape[1], axis=1)
        else:
            outdata[:] = chunk[:, : outdata.shape[1]]

        if not self.is_playing:
            import sounddevice as sd
            raise sd.CallbackStop()

    def get_next_playback_block(self, frames: int = 1024) -> np.ndarray:
        if self._play_buffer.size == 0:
            self.prepare_playback_buffer()
        total = self._play_buffer.shape[0]
        out_ch = self._play_buffer.shape[1] if self._play_buffer.ndim == 2 else self.channels
        if total == 0:
            self.is_playing = False
            return np.zeros((frames, out_ch), dtype=np.float32)

        loop_s, loop_e = (0, total)
        if self.loop_enabled and self.loop_range_frames is not None:
            loop_s = max(0, min(total - 1, self.loop_range_frames[0]))
            loop_e = max(loop_s + 1, min(total, self.loop_range_frames[1]))

        end_pos = self._play_cursor + frames
        if end_pos <= loop_e:
            chunk = self._play_buffer[self._play_cursor : end_pos]
            self._play_cursor = end_pos
        else:
            part1 = self._play_buffer[self._play_cursor : loop_e]
            rem = frames - len(part1)
            if self.loop_enabled:
                self._play_cursor = loop_s + min(rem, loop_e - loop_s)
                part2 = self._play_buffer[loop_s : self._play_cursor]
            else:
                self.is_playing = False
                self._play_cursor = 0
                part2 = np.zeros((rem, out_ch), dtype=np.float32)
            chunk = np.vstack([part1, part2]) if len(part1) > 0 else part2

        self._update_telemetry(chunk, frame_pos=self._play_cursor)
        return chunk

    def _update_telemetry(self, block: np.ndarray, frame_pos: int) -> None:
        if block.size == 0:
            return
        ch0 = block[:, 0]
        ch1 = block[:, 1] if block.shape[1] > 1 else ch0

        peak_l = float(np.max(np.abs(ch0)))
        peak_r = float(np.max(np.abs(ch1)))
        rms_l = float(np.sqrt(np.mean(ch0 ** 2) + 1e-12))
        rms_r = float(np.sqrt(np.mean(ch1 ** 2) + 1e-12))

        denom = float(np.sqrt(np.sum(ch0 ** 2) * np.sum(ch1 ** 2)) + 1e-12)
        corr = float(np.sum(ch0 * ch1) / denom)

        self.telemetry = AudioTelemetry(
            peak_db_l=20.0 * np.log10(max(peak_l, 1e-5)),
            peak_db_r=20.0 * np.log10(max(peak_r, 1e-5)),
            rms_db_l=20.0 * np.log10(max(rms_l, 1e-5)),
            rms_db_r=20.0 * np.log10(max(rms_r, 1e-5)),
            phase_correlation=float(np.clip(corr, -1.0, 1.0)),
            current_frame=frame_pos,
            current_time_sec=frame_pos / max(1, self.sample_rate),
            is_clipping=(peak_l >= 0.985 or peak_r >= 0.985),
            feedback_warning=self.telemetry.feedback_warning,
        )