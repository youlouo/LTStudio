# project/edl.py
from __future__ import annotations
import copy
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class ClipSlice:
    """
    Неразрушающий аудио-объект на многодорожечном таймлайне.
    Его можно двигать во времени (timeline_start_frame), переносить между дорожками (track_index),
    разделять (Split) и менять ему громкость (gain_db).
    """
    clip_id: str
    source_id: str
    source_start_frame: int
    frame_count: int
    timeline_start_frame: int = 0
    track_index: int = 0
    gain_db: float = 0.0
    name: str = "Clip"
    color: str = "#1E88E5"
    fade_in_frames: int = 0
    fade_out_frames: int = 0
    crossfade_with_prev_frames: int = 0

    @property
    def timeline_end_frame(self) -> int:
        return self.timeline_start_frame + self.frame_count


@dataclass
class AudioTrack:
    """Отдельная аудиодорожка для наложения записей друг на друга."""
    track_id: str
    name: str
    gain_db: float = 0.0
    muted: bool = False
    solo: bool = False


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
    """
    Многодорожечный неразрушающий таймлайн (Multitrack EDL).
    Поддерживает наложение дорожек, разделение клипов (Split), свободное перетаскивание
    объектов мышью, регулировку громкости отдельных клипов, выделенных участков и дорожек.
    """

    TRACK_COLORS = ["#1E88E5", "#43A047", "#8E24AA", "#FB8C00", "#00ACC1", "#E53935"]

    def __init__(self, sample_rate: int = 48000, channels: int = 1) -> None:
        self.sample_rate: int = sample_rate
        self.channels: int = channels
        self._sources: Dict[str, np.ndarray] = {}
        self.tracks: List[AudioTrack] = [
            AudioTrack(track_id="trk_1", name="Дорожка 1 (Основная)"),
            AudioTrack(track_id="trk_2", name="Дорожка 2 (Наложение / Фоновая)"),
            AudioTrack(track_id="trk_3", name="Дорожка 3 (Доп. голос / Эффекты)"),
        ]
        self.clips: List[ClipSlice] = []
        self.markers: List[TimelineMarker] = []
        self.regions: List[TimelineRegion] = []
        self._clipboard: List[ClipSlice] = []

        self.history_nodes: Dict[str, HistoryNode] = {}
        self.current_node_id: Optional[str] = None
        self.snapshots: Dict[str, str] = {}
        self._commit_state("Инициализация проекта")

    def add_track(self, name: Optional[str] = None) -> AudioTrack:
        idx = len(self.tracks) + 1
        trk = AudioTrack(track_id=f"trk_{uuid.uuid4().hex[:6]}", name=name or f"Дорожка {idx}")
        self.tracks.append(trk)
        self._commit_state(f"Добавлена {trk.name}")
        return trk

    def set_track_gain(self, track_index: int, gain_db: float) -> None:
        if 0 <= track_index < len(self.tracks):
            self.tracks[track_index].gain_db = float(np.clip(gain_db, -60.0, 24.0))
            self._commit_state(f"Громкость дорожки {track_index + 1}: {gain_db:+.1f} дБ")

    def toggle_track_mute(self, track_index: int) -> bool:
        if 0 <= track_index < len(self.tracks):
            self.tracks[track_index].muted = not self.tracks[track_index].muted
            self._commit_state(f"Mute дорожки {track_index + 1}")
            return self.tracks[track_index].muted
        return False

    def register_source(self, source_id: str, audio: np.ndarray, append_to_timeline: bool = True,
                        track_index: int = 0, start_frame: Optional[int] = None) -> ClipSlice | None:
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        immutable = audio.astype(np.float32, copy=True)
        immutable.flags.writeable = False
        self._sources[source_id] = immutable
        self.channels = max(self.channels, immutable.shape[1])

        while len(self.tracks) <= track_index:
            self.add_track()

        if append_to_timeline:
            if start_frame is None:
                # Ставим в конец выбранной дорожки
                track_clips = [c for c in self.clips if c.track_index == track_index]
                start_frame = max((c.timeline_end_frame for c in track_clips), default=0)

            color = self.TRACK_COLORS[track_index % len(self.TRACK_COLORS)]
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
            self._commit_state(f"Добавлен клип {source_id} на дорожку {track_index + 1}")
            return new_clip
        return None

    def get_source_preview(self, source_id: str, max_bins: int = 300) -> np.ndarray:
        """Возвращает уменьшенную огибающую клипа для быстрой отрисовки внутри прямоугольника."""
        if source_id not in self._sources:
            return np.zeros(max_bins, dtype=np.float32)
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
    # Объектные операции: Разделение (Split), Перемещение (Move), Громкость
    # ------------------------------------------------------------------

    def split_clip_at(self, frame: int, clip_id: Optional[str] = None,
                      track_index: Optional[int] = None) -> bool:
        """Разрезает клип в позиции курсора (клавиша S) на два независимых объекта."""
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
            left_len = frame - target.timeline_start_frame
            right_len = target.frame_count - left_len

            right_clip = copy.deepcopy(target)
            right_clip.clip_id = uuid.uuid4().hex[:8]
            right_clip.source_start_frame = target.source_start_frame + left_len
            right_clip.frame_count = right_len
            right_clip.timeline_start_frame = frame
            right_clip.fade_in_frames = min(128, right_len // 4)

            target.frame_count = left_len
            target.fade_out_frames = min(128, left_len // 4)
            self.clips.append(right_clip)

        self._commit_state("Разделение клипа (Split)")
        return True

    def move_clip(self, clip_id: str, new_timeline_start: int, new_track_index: int) -> bool:
        """Перемещает аудио-объект по времени и между дорожками."""
        for c in self.clips:
            if c.clip_id == clip_id:
                while len(self.tracks) <= new_track_index:
                    self.add_track()
                c.timeline_start_frame = max(0, int(new_timeline_start))
                c.track_index = max(0, int(new_track_index))
                c.color = self.TRACK_COLORS[c.track_index % len(self.TRACK_COLORS)]
                self._commit_state(f"Перемещение клипа {c.name}")
                return True
        return False

    def set_clip_gain(self, clip_id: str, gain_db: float) -> bool:
        """Устанавливает индивидуальное усиление (громкость) выбранного клипа."""
        for c in self.clips:
            if c.clip_id == clip_id:
                c.gain_db = float(np.clip(gain_db, -60.0, 36.0))
                self._commit_state(f"Громкость клипа {c.name}: {c.gain_db:+.1f} дБ")
                return True
        return False

    def delete_clip(self, clip_id: str) -> bool:
        before = len(self.clips)
        self.clips = [c for c in self.clips if c.clip_id != clip_id]
        if len(self.clips) < before:
            self._commit_state("Удаление клипа")
            return True
        return False

    def apply_gain_to_range(self, start_frame: int, end_frame: int, delta_db: float,
                            track_index: Optional[int] = None) -> None:
        """
        Если выделен отрезок внутри клипа, автоматически разрезает клип по границам отрезка
        и увеличивает/уменьшает громкость только этого отрезка!
        """
        if end_frame <= start_frame:
            return
        self.split_clip_at(start_frame, track_index=track_index)
        self.split_clip_at(end_frame, track_index=track_index)

        for c in self.clips:
            if track_index is not None and c.track_index != track_index:
                continue
            if c.timeline_start_frame >= start_frame - 16 and c.timeline_end_frame <= end_frame + 16:
                c.gain_db = float(np.clip(c.gain_db + delta_db, -60.0, 36.0))
        self._commit_state(f"Громкость отрезка ({delta_db:+.1f} дБ)")

    # ------------------------------------------------------------------
    # Совместимость с классическими операциями (Cut / Delete / Transcript)
    # ------------------------------------------------------------------

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
    # Многодорожечный рендеринг (суммирование и наложение всех дорожек)
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
            seg_len = ov_end - ov_start
            piece = raw[clip_offset : clip_offset + seg_len]

            dest_start = ov_start - start_frame
            dest_end = dest_start + len(piece)
            mix_bus[dest_start:dest_end] += piece

        # Защита мастер-шины от жесткого цифрового перегруза при сложении нескольких дорожек
        peak = float(np.max(np.abs(mix_bus))) if mix_bus.size > 0 else 0.0
        if peak > 0.99:
            mix_bus = np.tanh(mix_bus)
        return mix_bus

    def _materialize_clip(self, clip: ClipSlice, total_gain_db: float = 0.0) -> np.ndarray:
        if clip.source_id == "__SILENCE__" or clip.source_id not in self._sources:
            buf = np.zeros((clip.frame_count, self.channels), dtype=np.float32)
        else:
            src = self._sources[clip.source_id]
            s = clip.source_start_frame
            e = min(src.shape[0], s + clip.frame_count)
            buf = src[s:e].copy()
            if buf.shape[1] < self.channels:
                buf = np.repeat(buf, self.channels, axis=1)
            elif buf.shape[1] > self.channels:
                buf = buf[:, : self.channels]

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
    # Дерево истории (Undo / Redo / Снапшоты)
    # ------------------------------------------------------------------

    def _commit_state(self, action_label: str) -> str:
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
        return node_id

    def undo(self) -> bool:
        if not self.current_node_id:
            return False
        curr = self.history_nodes[self.current_node_id]
        if not curr.parent_id or curr.parent_id not in self.history_nodes:
            return False
        self._restore_node(curr.parent_id)
        return True

    def redo(self, branch_index: int = -1) -> bool:
        if not self.current_node_id:
            return False
        curr = self.history_nodes[self.current_node_id]
        if not curr.children_ids:
            return False
        self._restore_node(curr.children_ids[branch_index])
        return True

    def create_snapshot(self, name: str) -> str:
        node_id = self._commit_state(f"Снапшот: {name}")
        self.snapshots[name] = node_id
        return node_id

    def restore_snapshot(self, name: str) -> bool:
        if name not in self.snapshots:
            return False
        self._restore_node(self.snapshots[name])
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
            "tracks": [asdict(t) for t in self.tracks],
            "clips": [asdict(c) for c in self.clips],
            "markers": [asdict(m) for m in self.markers],
            "regions": [asdict(r) for r in self.regions],
            "snapshots": self.snapshots,
        }