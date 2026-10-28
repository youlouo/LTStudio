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
    """Неразрушающий срез исходного аудио-буфера (или виртуальной тишины)."""
    clip_id: str
    source_id: str              # ID исходного файла или "__SILENCE__"
    source_start_frame: int
    frame_count: int
    gain_db: float = 0.0
    fade_in_frames: int = 0
    fade_out_frames: int = 0
    crossfade_with_prev_frames: int = 0


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
    """Узел дерева истории (поддерживает ветвление при новых действиях после Undo)."""
    node_id: str
    action_label: str
    timestamp: float
    clips_state: List[ClipSlice]
    markers_state: List[TimelineMarker]
    regions_state: List[TimelineRegion]
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)


class NonDestructiveTimeline:
    """
    Движок неразрушающего монтажа (EDL).
    Все операции (Cut, Copy, Paste, Trim, Fade, Crossfade, Silence Insert, Delete by Transcript)
    мгновенно модифицируют только метаданные срезов и сохраняются в дерево истории.
    """

    def __init__(self, sample_rate: int = 48000, channels: int = 1) -> None:
        self.sample_rate: int = sample_rate
        self.channels: int = channels
        self._sources: Dict[str, np.ndarray] = {}
        self.clips: List[ClipSlice] = []
        self.markers: List[TimelineMarker] = []
        self.regions: List[TimelineRegion] = []
        self._clipboard: List[ClipSlice] = []

        # Дерево истории (Branching Undo/Redo) и именованные снапшоты
        self.history_nodes: Dict[str, HistoryNode] = {}
        self.current_node_id: Optional[str] = None
        self.snapshots: Dict[str, str] = {}  # snapshot_name -> node_id
        self._commit_state("Инициализация проекта")

    def register_source(self, source_id: str, audio: np.ndarray, append_to_timeline: bool = True) -> None:
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        # Храним исходник в неизменяемом виде (read-only view)
        immutable = audio.astype(np.float32, copy=True)
        immutable.flags.writeable = False
        self._sources[source_id] = immutable
        self.channels = immutable.shape[1]

        if append_to_timeline:
            self.clips.append(
                ClipSlice(
                    clip_id=uuid.uuid4().hex[:8],
                    source_id=source_id,
                    source_start_frame=0,
                    frame_count=immutable.shape[0],
                )
            )
            self._commit_state(f"Импорт источника {source_id}")

    @property
    def total_frames(self) -> int:
        total = 0
        for i, c in enumerate(self.clips):
            total += c.frame_count
            if i > 0:
                total -= min(c.crossfade_with_prev_frames, c.frame_count)
        return max(0, total)

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / max(1, self.sample_rate)

    # ------------------------------------------------------------------
    # Базовые операции неразрушающего редактирования
    # ------------------------------------------------------------------

    def copy_range(self, start_frame: int, end_frame: int) -> None:
        self._clipboard = self._extract_subclips(start_frame, end_frame)

    def cut_range(self, start_frame: int, end_frame: int, crossfade_ms: float = 15.0) -> None:
        self.copy_range(start_frame, end_frame)
        self.delete_range(start_frame, end_frame, crossfade_ms=crossfade_ms, label="Вырезать (Cut)")

    def delete_range(self, start_frame: int, end_frame: int,
                     crossfade_ms: float = 15.0, label: str = "Удаление фрагмента") -> None:
        if end_frame <= start_frame or not self.clips:
            return
        left = self._extract_subclips(0, start_frame)
        right = self._extract_subclips(end_frame, self.total_frames)
        xfade_frames = int(crossfade_ms * 0.001 * self.sample_rate)
        if left and right and xfade_frames > 0:
            right[0].crossfade_with_prev_frames = min(xfade_frames, left[-1].frame_count // 2, right[0].frame_count // 2)
        self.clips = left + right
        self._commit_state(label)

    def trim_to_range(self, start_frame: int, end_frame: int) -> None:
        if end_frame <= start_frame:
            return
        self.clips = self._extract_subclips(start_frame, end_frame)
        self._commit_state("Обрезка по краям (Trim)")

    def paste_at(self, target_frame: int) -> None:
        if not self._clipboard:
            return
        left = self._extract_subclips(0, target_frame)
        right = self._extract_subclips(target_frame, self.total_frames)
        pasted = [copy.deepcopy(c) for c in self._clipboard]
        for c in pasted:
            c.clip_id = uuid.uuid4().hex[:8]
        self.clips = left + pasted + right
        self._commit_state("Вставка (Paste)")

    def insert_silence(self, at_frame: int, duration_sec: float) -> None:
        frames = int(round(duration_sec * self.sample_rate))
        if frames <= 0:
            return
        left = self._extract_subclips(0, at_frame)
        right = self._extract_subclips(at_frame, self.total_frames)
        silence_clip = ClipSlice(
            clip_id=uuid.uuid4().hex[:8],
            source_id="__SILENCE__",
            source_start_frame=0,
            frame_count=frames,
        )
        self.clips = left + [silence_clip] + right
        self._commit_state(f"Вставка тишины ({duration_sec:.2f} с)")

    def apply_fade(self, start_frame: int, end_frame: int, fade_in_ms: float = 0.0, fade_out_ms: float = 0.0) -> None:
        left = self._extract_subclips(0, start_frame)
        mid = self._extract_subclips(start_frame, end_frame)
        right = self._extract_subclips(end_frame, self.total_frames)
        if mid:
            if fade_in_ms > 0:
                mid[0].fade_in_frames = min(mid[0].frame_count, int(fade_in_ms * 0.001 * self.sample_rate))
            if fade_out_ms > 0:
                mid[-1].fade_out_frames = min(mid[-1].frame_count, int(fade_out_ms * 0.001 * self.sample_rate))
        self.clips = left + mid + right
        self._commit_state("Применение Fade In/Out")

    def apply_gain_to_range(self, start_frame: int, end_frame: int, delta_db: float) -> None:
        left = self._extract_subclips(0, start_frame)
        mid = self._extract_subclips(start_frame, end_frame)
        right = self._extract_subclips(end_frame, self.total_frames)
        for c in mid:
            c.gain_db = float(np.clip(c.gain_db + delta_db, -60.0, 36.0))
        self.clips = left + mid + right
        self._commit_state(f"Изменение уровня ({delta_db:+.1f} дБ)")

    def delete_transcript_words(self, word_frame_spans: List[Tuple[int, int]], crossfade_ms: float = 18.0) -> None:
        """
        Удаление выбранных слов/фраз по транскрипту в обратном хронологическом порядке
        (чтобы индексы ранних слов не смещались) с автоматической склейкой кроссфейдами.
        """
        sorted_spans = sorted(word_frame_spans, key=lambda x: x[0], reverse=True)
        for s_frame, e_frame in sorted_spans:
            if e_frame > s_frame:
                left = self._extract_subclips(0, s_frame)
                right = self._extract_subclips(e_frame, self.total_frames)
                xfade = int(crossfade_ms * 0.001 * self.sample_rate)
                if left and right and xfade > 0:
                    right[0].crossfade_with_prev_frames = min(
                        xfade, left[-1].frame_count // 2, right[0].frame_count // 2
                    )
                self.clips = left + right
        self._commit_state(f"Удаление слов по транскрипту ({len(sorted_spans)} шт.)")

    # ------------------------------------------------------------------
    # Рендеринг неразрушающего таймлайна в линейный буфер
    # ------------------------------------------------------------------

    def render_timeline(self, start_frame: int = 0, end_frame: Optional[int] = None) -> np.ndarray:
        """Собирает аудио из неразрушающих срезов с учетом Gain, Fade In/Out и Equal-Power Crossfades."""
        if not self.clips:
            return np.zeros((0, self.channels), dtype=np.float32)

        rendered_pieces: List[np.ndarray] = []
        for idx, clip in enumerate(self.clips):
            raw = self._materialize_clip(clip)
            xfade = min(clip.crossfade_with_prev_frames, len(raw)) if idx > 0 else 0
            if xfade > 0 and rendered_pieces and len(rendered_pieces[-1]) >= xfade:
                prev = rendered_pieces[-1]
                t = np.linspace(0.0, np.pi / 2.0, xfade, dtype=np.float32)[:, np.newaxis]
                mixed = prev[-xfade:] * np.cos(t) + raw[:xfade] * np.sin(t)
                rendered_pieces[-1] = prev[:-xfade]
                rendered_pieces.append(mixed)
                rendered_pieces.append(raw[xfade:])
            else:
                rendered_pieces.append(raw)

        full = np.vstack(rendered_pieces) if rendered_pieces else np.zeros((0, self.channels), dtype=np.float32)
        if end_frame is None:
            end_frame = full.shape[0]
        return full[max(0, start_frame) : min(full.shape[0], end_frame)]

    def _materialize_clip(self, clip: ClipSlice) -> np.ndarray:
        if clip.source_id == "__SILENCE__" or clip.source_id not in self._sources:
            buf = np.zeros((clip.frame_count, self.channels), dtype=np.float32)
        else:
            src = self._sources[clip.source_id]
            s = clip.source_start_frame
            e = min(src.shape[0], s + clip.frame_count)
            buf = src[s:e].copy()

        if abs(clip.gain_db) > 1e-4:
            buf *= float(10.0 ** (clip.gain_db / 20.0))

        n = buf.shape[0]
        if clip.fade_in_frames > 0 and n > 0:
            fi = min(n, clip.fade_in_frames)
            buf[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, np.newaxis]
        if clip.fade_out_frames > 0 and n > 0:
            fo = min(n, clip.fade_out_frames)
            buf[n - fo :] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)[:, np.newaxis]
        return buf

    def _extract_subclips(self, range_start: int, range_end: int) -> List[ClipSlice]:
        """Точное вычисление подграфа клипов, попадающих в интервал [range_start, range_end)."""
        result: List[ClipSlice] = []
        cursor = 0
        for idx, clip in enumerate(self.clips):
            xfade = min(clip.crossfade_with_prev_frames, clip.frame_count) if idx > 0 else 0
            clip_start = cursor - xfade
            clip_end = clip_start + clip.frame_count
            cursor = clip_end

            overlap_start = max(range_start, clip_start)
            overlap_end = min(range_end, clip_end)
            if overlap_end > overlap_start:
                offset = overlap_start - clip_start
                count = overlap_end - overlap_start
                sub = copy.deepcopy(clip)
                sub.clip_id = uuid.uuid4().hex[:8]
                sub.source_start_frame = clip.source_start_frame + offset
                sub.frame_count = count
                sub.crossfade_with_prev_frames = 0
                if offset > 0:
                    sub.fade_in_frames = 0
                if overlap_end < clip_end:
                    sub.fade_out_frames = 0
                result.append(sub)
        return result

    # ------------------------------------------------------------------
    # Дерево истории (Undo / Redo / Ветки / Снапшоты)
    # ------------------------------------------------------------------

    def _commit_state(self, action_label: str) -> str:
        node_id = uuid.uuid4().hex[:10]
        node = HistoryNode(
            node_id=node_id,
            action_label=action_label,
            timestamp=time.time(),
            clips_state=copy.deepcopy(self.clips),
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
        target_id = curr.children_ids[branch_index]
        self._restore_node(target_id)
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
        self.markers = copy.deepcopy(node.markers_state)
        self.regions = copy.deepcopy(node.regions_state)
        self.current_node_id = node_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "clips": [asdict(c) for c in self.clips],
            "markers": [asdict(m) for m in self.markers],
            "regions": [asdict(r) for r in self.regions],
            "snapshots": self.snapshots,
        }