# project/edl.py
from __future__ import annotations
import copy
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
import soundfile as sf
from scipy import signal


@dataclass
class ClipSlice:
    clip_id: str
    source_id: str
    source_start_frame: int
    frame_count: int
    timeline_start_frame: int = 0
    track_index: int = 0
    gain_db: float = 0.0
    speed_rate: float = 1.0
    name: str = "Clip"
    color: str = "#1E88E5"
    fade_in_frames: int = 0
    fade_out_frames: int = 0
    crossfade_with_prev_frames: int = 0
    dsp_profile_name: str = ""
    dsp_chain_data: Optional[Dict[str, Any]] = None

    @property
    def effective_frame_count(self) -> int:
        rate = max(0.25, min(4.0, float(self.speed_rate)))
        return max(1, int(round(self.frame_count / rate)))

    @property
    def timeline_end_frame(self) -> int:
        return self.timeline_start_frame + self.effective_frame_count


@dataclass
class AudioTrack:
    track_id: str
    name: str
    gain_db: float = 0.0
    muted: bool = False
    solo: bool = False
    color: str = "#1E88E5"


@dataclass
class TimelineMarker:
    marker_id: str
    name: str
    frame: int
    color: str = "#FFB300"


@dataclass
class TimelineRegion:
    region_id: str
    name: str
    start_frame: int
    end_frame: int
    color: str = "#00BCD4"


@dataclass
class HistoryNode:
    node_id: str
    action_label: str
    timestamp: float
    clips_state: List[ClipSlice]
    tracks_state: List[AudioTrack]
    markers_state: List[TimelineMarker]
    regions_state: List[TimelineRegion]
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)


class NonDestructiveTimeline:
    """Многодорожечный таймлайн с управлением маркерами, проверкой доступности Undo/Redo и сдвигом по дорожкам."""

    TRACK_COLORS = ["#1E88E5", "#43A047", "#8E24AA", "#FB8C00", "#00ACC1", "#E53935"]

    def __init__(self, sample_rate: int = 48000, channels: int = 1) -> None:
        self.sample_rate: int = sample_rate
        self.channels: int = channels
        self._sources: Dict[str, np.ndarray] = {}
        self.source_files: Dict[str, str] = {}
        self._clip_dsp_cache: Dict[str, Tuple[Any, np.ndarray]] = {}

        self.tracks: List[AudioTrack] = [
            AudioTrack(track_id="trk_1", name="Дорожка 1 (Основная)", color="#1E88E5"),
            AudioTrack(track_id="trk_2", name="Дорожка 2 (Наложение)", color="#43A047"),
            AudioTrack(track_id="trk_3", name="Дорожка 3 (Фон / Эффекты)", color="#8E24AA"),
        ]
        self.clips: List[ClipSlice] = []
        self.markers: List[TimelineMarker] = []
        self.regions: List[TimelineRegion] = []

        self.history_nodes: Dict[str, HistoryNode] = {}
        self.current_node_id: Optional[str] = None
        self.snapshots: Dict[str, str] = {}
        self.is_dirty: bool = False
        self._commit_state("Инициализация проекта", mark_dirty=False)

    @property
    def can_undo(self) -> bool:
        if not self.current_node_id or self.current_node_id not in self.history_nodes:
            return False
        parent = self.history_nodes[self.current_node_id].parent_id
        return bool(parent and parent in self.history_nodes)

    @property
    def can_redo(self) -> bool:
        if not self.current_node_id or self.current_node_id not in self.history_nodes:
            return False
        return len(self.history_nodes[self.current_node_id].children_ids) > 0

    # ------------------------------------------------------------------
    # Управление маркерами (Добавление, Перемещение, Переименование, Удаление)
    # ------------------------------------------------------------------

    def add_marker(self, frame: int, name: Optional[str] = None) -> TimelineMarker:
        idx = len(self.markers) + 1
        m = TimelineMarker(
            marker_id=f"m_{uuid.uuid4().hex[:6]}",
            name=name or f"M{idx}",
            frame=max(0, int(frame)),
        )
        self.markers.append(m)
        self._commit_state(f"Добавлен маркер {m.name}")
        return m

    def move_marker(self, marker_id: str, new_frame: int) -> bool:
        for m in self.markers:
            if m.marker_id == marker_id:
                m.frame = max(0, int(new_frame))
                self._commit_state(f"Перемещен маркер {m.name}")
                return True
        return False

    def rename_marker(self, marker_id: str, new_name: str) -> bool:
        for m in self.markers:
            if m.marker_id == marker_id and new_name.strip():
                m.name = new_name.strip()
                self._commit_state(f"Переименован маркер в {m.name}")
                return True
        return False

    def delete_marker(self, marker_id: str) -> bool:
        before = len(self.markers)
        self.markers = [m for m in self.markers if m.marker_id != marker_id]
        if len(self.markers) < before:
            self._commit_state("Удален маркер")
            return True
        return False

    def clear_all_markers(self) -> bool:
        if not self.markers:
            return False
        self.markers.clear()
        self._commit_state("Удалены все маркеры")
        return True

    # ------------------------------------------------------------------
    # Дорожки и Источники
    # ------------------------------------------------------------------

    def add_track(self, name: Optional[str] = None, color: Optional[str] = None) -> AudioTrack:
        idx = len(self.tracks) + 1
        col = color or self.TRACK_COLORS[(idx - 1) % len(self.TRACK_COLORS)]
        trk = AudioTrack(track_id=f"trk_{uuid.uuid4().hex[:6]}", name=name or f"Дорожка {idx}", color=col)
        self.tracks.append(trk)
        self._commit_state(f"Добавлена {trk.name}")
        return trk

    def set_track_color(self, track_index: int, color_hex: str) -> None:
        if 0 <= track_index < len(self.tracks):
            self.tracks[track_index].color = color_hex
            for c in self.clips:
                if c.track_index == track_index:
                    c.color = color_hex
            self._commit_state(f"Изменен цвет дорожки {track_index + 1}")

    def set_clips_color(self, clip_ids: Set[str], color_hex: str) -> None:
        changed = False
        for c in self.clips:
            if c.clip_id in clip_ids:
                c.color = color_hex
                changed = True
        if changed:
            self._commit_state("Изменен цвет выбранных клипов")

    def set_track_gain(self, track_index: int, gain_db: float) -> None:
        if 0 <= track_index < len(self.tracks):
            self.tracks[track_index].gain_db = float(np.clip(gain_db, -60.0, 24.0))
            self._commit_state(f"Громкость дорожки {track_index + 1}: {gain_db:+.1f} дБ")

    def toggle_track_mute(self, track_index: int) -> bool:
        if 0 <= track_index < len(self.tracks):
            self.tracks[track_index].muted = not self.tracks[track_index].muted
            self._commit_state(f"Приглушение дорожки {track_index + 1}")
            return self.tracks[track_index].muted
        return False

    def register_source(self, source_id: str, audio: np.ndarray, append_to_timeline: bool = True,
                        track_index: int = 0, start_frame: Optional[int] = None,
                        file_path: Optional[str] = None) -> Optional[ClipSlice]:
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        immutable = audio.astype(np.float32, copy=True)
        immutable.flags.writeable = False
        self._sources[source_id] = immutable
        if file_path:
            self.source_files[source_id] = str(file_path)
        self.channels = max(self.channels, immutable.shape[1])

        while len(self.tracks) <= track_index:
            self.add_track()

        if append_to_timeline:
            if start_frame is None:
                track_clips = [c for c in self.clips if c.track_index == track_index]
                start_frame = max((c.timeline_end_frame for c in track_clips), default=0)

            color = self.tracks[track_index].color
            new_clip = ClipSlice(
                clip_id=uuid.uuid4().hex[:8],
                source_id=source_id,
                source_start_frame=0,
                frame_count=immutable.shape[0],
                timeline_start_frame=max(0, int(start_frame)),
                track_index=track_index,
                name=source_id,
                color=color,
            )
            self.clips.append(new_clip)
            self._commit_state(f"Добавлен клип {source_id}")
            return new_clip
        return None

    def get_source_preview(self, source_id: str) -> np.ndarray:
        if source_id not in self._sources:
            return np.zeros(0, dtype=np.float32)
        return self._sources[source_id][:, 0]

    @property
    def total_frames(self) -> int:
        if not self.clips:
            return 0
        return max(c.timeline_end_frame for c in self.clips)

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / max(1, self.sample_rate)

    # ------------------------------------------------------------------
    # Одиночные и групповые операции над клипами (Сдвиг по времени и дорожкам, Слияние, Фейды)
    # ------------------------------------------------------------------

    def set_clips_fades(self, clip_ids: Set[str], fade_in_sec: Optional[float] = None,
                        fade_out_sec: Optional[float] = None) -> bool:
        changed = False
        for c in self.clips:
            if c.clip_id in clip_ids:
                if fade_in_sec is not None:
                    fi_frames = int(round(max(0.0, fade_in_sec) * self.sample_rate))
                    c.fade_in_frames = min(c.effective_frame_count, fi_frames)
                if fade_out_sec is not None:
                    fo_frames = int(round(max(0.0, fade_out_sec) * self.sample_rate))
                    c.fade_out_frames = min(c.effective_frame_count, fo_frames)
                changed = True
        if changed:
            self._commit_state("Изменение проявления/затухания")
        return changed

    def set_clips_speed(self, clip_ids: Set[str], speed_rate: float) -> bool:
        rate = float(np.clip(speed_rate, 0.5, 2.0))
        changed = False
        for c in self.clips:
            if c.clip_id in clip_ids:
                c.speed_rate = rate
                self._clip_dsp_cache.pop(c.clip_id, None)
                changed = True
        if changed:
            self._commit_state(f"Скорость клипов: {rate}x")
        return changed

    def set_clips_dsp_profile(self, clip_ids: Set[str], profile_name: str,
                              dsp_chain_dict: Optional[Dict[str, Any]]) -> bool:
        changed = False
        for c in self.clips:
            if c.clip_id in clip_ids:
                c.dsp_profile_name = profile_name
                c.dsp_chain_data = copy.deepcopy(dsp_chain_dict) if dsp_chain_dict else None
                self._clip_dsp_cache.pop(c.clip_id, None)
                changed = True
        if changed:
            label = f"Профиль «{profile_name}» на клипы" if profile_name else "Сброс профиля клипов"
            self._commit_state(label)
        return changed

    def nudge_clips(self, clip_ids: Set[str], delta_frames: int = 0, delta_track: int = 0) -> bool:
        """Сдвигает выделенные клипы влево/вправо по времени и вверх/вниз между дорожками (стрелки клавиатуры)."""
        if not clip_ids or (delta_frames == 0 and delta_track == 0):
            return False
        changed = False
        max_trk = max(0, len(self.tracks) - 1)
        for c in self.clips:
            if c.clip_id in clip_ids:
                if delta_frames != 0:
                    c.timeline_start_frame = max(0, c.timeline_start_frame + int(delta_frames))
                    changed = True
                if delta_track != 0:
                    new_trk = max(0, min(max_trk, c.track_index + int(delta_track)))
                    if new_trk != c.track_index:
                        c.track_index = new_trk
                        changed = True
        if changed:
            self._commit_state("Перемещение клипов клавишами")
        return changed

    def merge_clips(self, clip_ids: Set[str], save_dir: Optional[Path] = None) -> Optional[ClipSlice]:
        selected = [c for c in self.clips if c.clip_id in clip_ids]
        if len(selected) < 2:
            return None

        selected.sort(key=lambda x: x.timeline_start_frame)
        min_start = min(c.timeline_start_frame for c in selected)
        max_end = max(c.timeline_end_frame for c in selected)
        total_len = max_end - min_start
        if total_len <= 0:
            return None

        target_track = selected[0].track_index
        target_color = selected[0].color
        merged_buf = np.zeros((total_len, self.channels), dtype=np.float32)

        for clip in selected:
            raw = self._materialize_clip(clip, total_gain_db=clip.gain_db)
            offset = clip.timeline_start_frame - min_start
            end_pos = min(total_len, offset + len(raw))
            seg_len = end_pos - offset
            if seg_len > 0:
                merged_buf[offset:end_pos] += raw[:seg_len]

        peak = float(np.max(np.abs(merged_buf))) if merged_buf.size > 0 else 0.0
        if peak > 0.99:
            merged_buf = np.tanh(merged_buf).astype(np.float32)

        merged_id = f"merged_{uuid.uuid4().hex[:6]}"
        file_path_str: Optional[str] = None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            out_wav = save_dir / f"{merged_id}.wav"
            sf.write(str(out_wav), merged_buf, self.sample_rate, subtype="PCM_24")
            file_path_str = str(out_wav)

        immutable = merged_buf.astype(np.float32, copy=True)
        immutable.flags.writeable = False
        self._sources[merged_id] = immutable
        if file_path_str:
            self.source_files[merged_id] = file_path_str

        self.clips = [c for c in self.clips if c.clip_id not in clip_ids]
        new_clip = ClipSlice(
            clip_id=uuid.uuid4().hex[:8],
            source_id=merged_id,
            source_start_frame=0,
            frame_count=immutable.shape[0],
            timeline_start_frame=min_start,
            track_index=target_track,
            name=f"Слияние ({len(selected)} шт.)",
            color=target_color,
        )
        self.clips.append(new_clip)
        self._commit_state(f"Объединение {len(selected)} клипов")
        return new_clip

    def apply_fade(self, start_frame: int, end_frame: int, fade_in_ms: float = 0.0, fade_out_ms: float = 0.0) -> None:
        if end_frame <= start_frame:
            return
        for c in self.clips:
            if c.timeline_start_frame >= start_frame - 16 and c.timeline_end_frame <= end_frame + 16:
                c.fade_in_frames = min(c.effective_frame_count, int(fade_in_ms * 0.001 * self.sample_rate))
                c.fade_out_frames = min(c.effective_frame_count, int(fade_out_ms * 0.001 * self.sample_rate))
        self._commit_state("Применение фейдов к выделенной зоне")

    def trim_to_range(self, start_frame: int, end_frame: int) -> None:
        if end_frame <= start_frame:
            return
        self.split_clip_at(start_frame)
        self.split_clip_at(end_frame)
        kept: List[ClipSlice] = []
        for c in self.clips:
            if c.timeline_start_frame >= start_frame - 16 and c.timeline_end_frame <= end_frame + 16:
                c.timeline_start_frame = max(0, c.timeline_start_frame - start_frame)
                kept.append(c)
        self.clips = kept
        self._commit_state("Обрезка за пределами выделенной зоны")

    def duplicate_clips(self, clip_ids: Set[str]) -> List[ClipSlice]:
        new_clips: List[ClipSlice] = []
        for c in list(self.clips):
            if c.clip_id in clip_ids:
                dup = copy.deepcopy(c)
                dup.clip_id = uuid.uuid4().hex[:8]
                dup.timeline_start_frame = c.timeline_end_frame + int(0.05 * self.sample_rate)
                self.clips.append(dup)
                new_clips.append(dup)
        if new_clips:
            self._commit_state(f"Дублирование клипов ({len(new_clips)} шт.)")
        return new_clips

    def split_clip_at(self, frame: int, clip_id: Optional[str] = None,
                      track_index: Optional[int] = None) -> bool:
        targets: List[ClipSlice] = []
        for c in self.clips:
            if clip_id and c.clip_id == clip_id:
                targets = [c]
                break
            if clip_id is None and (track_index is None or c.track_index == track_index):
                if c.timeline_start_frame + 64 < frame < c.timeline_end_frame - 64:
                    targets.append(c)

        if not targets:
            return False

        for target in targets:
            if not (target.timeline_start_frame + 32 < frame < target.timeline_end_frame - 32):
                continue
            rate = max(0.25, min(4.0, float(target.speed_rate)))
            timeline_left = frame - target.timeline_start_frame
            src_left = int(round(timeline_left * rate))
            src_right = max(1, target.frame_count - src_left)

            right_clip = copy.deepcopy(target)
            right_clip.clip_id = uuid.uuid4().hex[:8]
            right_clip.source_start_frame = target.source_start_frame + src_left
            right_clip.frame_count = src_right
            right_clip.timeline_start_frame = frame
            right_clip.fade_in_frames = min(128, right_clip.effective_frame_count // 4)

            target.frame_count = max(1, src_left)
            target.fade_out_frames = min(128, target.effective_frame_count // 4)
            self.clips.append(right_clip)

        self._commit_state("Разделение клипа")
        return True

    def commit_Bulk_move(self) -> None:
        self._commit_state("Перемещение клипов")

    def move_clip(self, clip_id: str, new_timeline_start: int, new_track_index: int) -> bool:
        for c in self.clips:
            if c.clip_id == clip_id:
                while len(self.tracks) <= new_track_index:
                    self.add_track()
                c.timeline_start_frame = max(0, int(new_timeline_start))
                c.track_index = max(0, int(new_track_index))
                self._commit_state(f"Перемещение клипа {c.name}")
                return True
        return False

    def set_clips_gain(self, clip_ids: Set[str], gain_db: float) -> bool:
        changed = False
        for c in self.clips:
            if c.clip_id in clip_ids:
                c.gain_db = float(np.clip(gain_db, -60.0, 36.0))
                changed = True
        if changed:
            self._commit_state(f"Громкость клипов: {gain_db:+.1f} дБ")
        return changed

    def delete_clips(self, clip_ids: Set[str]) -> bool:
        before = len(self.clips)
        self.clips = [c for c in self.clips if c.clip_id not in clip_ids]
        if len(self.clips) < before:
            self._commit_state(f"Удаление клипов ({before - len(self.clips)} шт.)")
            return True
        return False

    def cut_range(self, start_frame: int, end_frame: int, crossfade_ms: float = 15.0) -> None:
        self.delete_range(start_frame, end_frame, crossfade_ms=crossfade_ms, label="Вырезать участок")

    def delete_range(self, start_frame: int, end_frame: int,
                     crossfade_ms: float = 15.0, label: str = "Удаление фрагмента") -> None:
        if end_frame <= start_frame or not self.clips:
            return
        span = end_frame - start_frame
        self.split_clip_at(start_frame)
        self.split_clip_at(end_frame)

        remaining: List[ClipSlice] = []
        for c in self.clips:
            if c.timeline_start_frame >= start_frame - 8 and c.timeline_end_frame <= end_frame + 8:
                continue
            if c.timeline_start_frame >= end_frame - 8:
                c.timeline_start_frame = max(0, c.timeline_start_frame - span)
            remaining.append(c)
        self.clips = remaining
        self._commit_state(label)

    def delete_transcript_words(self, word_frame_spans: List[Tuple[int, int]], crossfade_ms: float = 18.0) -> None:
        sorted_spans = sorted(word_frame_spans, key=lambda x: x[0], reverse=True)
        for s_frame, e_frame in sorted_spans:
            if e_frame > s_frame:
                self.delete_range(s_frame, e_frame, crossfade_ms=crossfade_ms, label="Удаление слова")

    # ------------------------------------------------------------------
    # Рендеринг микса всех дорожек
    # ------------------------------------------------------------------

    def render_timeline(self, start_frame: int = 0, end_frame: Optional[int] = None) -> np.ndarray:
        total = self.total_frames
        if total == 0 or not self.clips:
            return np.zeros((0, self.channels), dtype=np.float32)

        if end_frame is None:
            end_frame = total
        end_frame = min(total, max(start_frame, end_frame))
        out_len = end_frame - start_frame
        if out_len <= 0:
            return np.zeros((0, self.channels), dtype=np.float32)

        mix_bus = np.zeros((out_len, self.channels), dtype=np.float32)
        any_solo = any(t.solo for t in self.tracks)

        for clip in self.clips:
            if clip.track_index < len(self.tracks):
                trk = self.tracks[clip.track_index]
                if trk.muted or (any_solo and not trk.solo):
                    continue
                trk_gain_db = trk.gain_db
            else:
                trk_gain_db = 0.0

            ov_start = max(start_frame, clip.timeline_start_frame)
            ov_end = min(end_frame, clip.timeline_end_frame)
            if ov_end <= ov_start:
                continue

            raw = self._materialize_clip(clip, total_gain_db=clip.gain_db + trk_gain_db)
            clip_offset = ov_start - clip.timeline_start_frame
            seg_len = min(ov_end - ov_start, max(0, len(raw) - clip_offset))
            if seg_len <= 0:
                continue
            piece = raw[clip_offset : clip_offset + seg_len]

            dest_start = ov_start - start_frame
            dest_end = dest_start + len(piece)
            mix_bus[dest_start:dest_end] += piece

        peak = float(np.max(np.abs(mix_bus))) if mix_bus.size > 0 else 0.0
        if peak > 0.99:
            mix_bus = np.tanh(mix_bus)
        return mix_bus

    def _materialize_clip(self, clip: ClipSlice, total_gain_db: float = 0.0) -> np.ndarray:
        if clip.source_id == "__SILENCE__" or clip.source_id not in self._sources:
            buf = np.zeros((clip.effective_frame_count, self.channels), dtype=np.float32)
        else:
            cache_key = (
                clip.source_id,
                clip.source_start_frame,
                clip.frame_count,
                round(clip.speed_rate, 3),
                clip.dsp_profile_name,
            )
            cached = self._clip_dsp_cache.get(clip.clip_id)
            if cached is not None and cached[0] == cache_key:
                buf = cached[1].copy()
            else:
                src = self._sources[clip.source_id]
                s = clip.source_start_frame
                e = min(src.shape[0], s + clip.frame_count)
                buf = src[s:e].copy()
                if buf.shape[1] < self.channels:
                    buf = np.repeat(buf, self.channels, axis=1)
                elif buf.shape[1] > self.channels:
                    buf = buf[:, : self.channels]

                rate = float(np.clip(clip.speed_rate, 0.5, 2.0))
                if abs(rate - 1.0) > 1e-2 and len(buf) > 16:
                    target_len = max(1, int(round(len(buf) / rate)))
                    buf = signal.resample(buf, target_len).astype(np.float32)

                if clip.dsp_chain_data:
                    from engine.dsp_chain import DSPEffectChain
                    clip_chain = DSPEffectChain.from_dict(clip.dsp_chain_data)
                    clip_chain.reset_all(self.sample_rate, self.channels)
                    buf = clip_chain.process(buf, self.sample_rate)

                self._clip_dsp_cache[clip.clip_id] = (cache_key, buf.copy())

        if abs(total_gain_db) > 1e-4:
            buf *= float(10.0 ** (total_gain_db / 20.0))

        n = buf.shape[0]
        if clip.fade_in_frames > 0 and n > 0:
            fi = min(n, clip.fade_in_frames)
            buf[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, np.newaxis]
        if clip.fade_out_frames > 0 and n > 0:
            fo = min(n, clip.fade_out_frames)
            buf[n - fo :] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)[:, np.newaxis]
        return buf

    # ------------------------------------------------------------------
    # История и сериализация проектов
    # ------------------------------------------------------------------

    def _commit_state(self, action_label: str, mark_dirty: bool = True) -> str:
        node_id = uuid.uuid4().hex[:10]
        node = HistoryNode(
            node_id=node_id,
            action_label=action_label,
            timestamp=time.time(),
            clips_state=copy.deepcopy(self.clips),
            tracks_state=copy.deepcopy(self.tracks),
            markers_state=copy.deepcopy(self.markers),
            regions_state=copy.deepcopy(self.regions),
            parent_id=self.current_node_id,
        )
        if self.current_node_id and self.current_node_id in self.history_nodes:
            self.history_nodes[self.current_node_id].children_ids.append(node_id)
        self.history_nodes[node_id] = node
        self.current_node_id = node_id
        if mark_dirty:
            self.is_dirty = True
        return node_id

    def undo(self) -> bool:
        if not self.can_undo:
            return False
        curr = self.history_nodes[self.current_node_id]
        self._restore_node(curr.parent_id)
        self.is_dirty = True
        return True

    def redo(self, branch_index: int = -1) -> bool:
        if not self.can_redo:
            return False
        curr = self.history_nodes[self.current_node_id]
        self._restore_node(curr.children_ids[branch_index])
        self.is_dirty = True
        return True

    def _restore_node(self, node_id: str) -> None:
        node = self.history_nodes[node_id]
        self.clips = copy.deepcopy(node.clips_state)
        self.tracks = copy.deepcopy(node.tracks_state)
        self.markers = copy.deepcopy(node.markers_state)
        self.regions = copy.deepcopy(node.regions_state)
        self.current_node_id = node_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "source_files": self.source_files,
            "tracks": [asdict(t) for t in self.tracks],
            "clips": [asdict(c) for c in self.clips],
            "markers": [asdict(m) for m in self.markers],
            "regions": [asdict(r) for r in self.regions],
            "snapshots": self.snapshots,
        }

    def load_from_dict(self, data: Dict[str, Any], fallback_dir: Optional[Path] = None) -> None:
        self.sample_rate = int(data.get("sample_rate", 48000))
        self.channels = int(data.get("channels", 1))
        self.source_files = dict(data.get("source_files", {}))
        self._sources.clear()
        self._clip_dsp_cache.clear()

        for src_id, fpath_str in list(self.source_files.items()):
            p = Path(fpath_str)
            if not p.exists() and fallback_dir is not None:
                alt = fallback_dir / p.name
                if alt.exists():
                    p = alt
            if p.exists():
                try:
                    audio, sr = sf.read(str(p), dtype="float32", always_2d=True)
                    if sr != self.sample_rate:
                        target_len = int(round(audio.shape[0] * self.sample_rate / sr))
                        audio = signal.resample(audio, target_len).astype(np.float32)
                    audio.flags.writeable = False
                    self._sources[src_id] = audio
                except Exception:
                    pass

        raw_tracks = data.get("tracks", [])
        if raw_tracks:
            self.tracks = [AudioTrack(**t) for t in raw_tracks]
        else:
            self.tracks = [
                AudioTrack(track_id="trk_1", name="Дорожка 1 (Основная)", color="#1E88E5"),
                AudioTrack(track_id="trk_2", name="Дорожка 2 (Наложение)", color="#43A047"),
                AudioTrack(track_id="trk_3", name="Дорожка 3 (Фон / Эффекты)", color="#8E24AA"),
            ]

        loaded_clips: List[ClipSlice] = []
        for c_dict in data.get("clips", []):
            if c_dict.get("source_id") in self._sources:
                c_dict.setdefault("speed_rate", 1.0)
                c_dict.setdefault("dsp_profile_name", "")
                c_dict.setdefault("dsp_chain_data", None)
                loaded_clips.append(ClipSlice(**c_dict))
        self.clips = loaded_clips
        self.markers = [TimelineMarker(**m) for m in data.get("markers", [])]
        self.regions = [TimelineRegion(**r) for r in data.get("regions", [])]
        self.history_nodes.clear()
        self.current_node_id = None
        self._commit_state("Загрузка проекта", mark_dirty=False)