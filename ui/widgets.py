# ui/widgets.py
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from scipy import signal
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QImage, QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QWidget

from project.edl import NonDestructiveTimeline, ClipSlice, TimelineMarker, TimelineRegion


class WaveformSpectrogramWidget(QWidget):
    """
    Многодорожечный DAW-холст LTStudio + Мини-карта + FFT Спектрограмма.
    Позволяет:
    - Кликать и перетаскивать отдельные аудио-клипы по времени и между дорожками;
    - Выделять отрезки (с зажатым Shift или на пустом месте) для изменения громкости/вырезания;
    - Масштабировать колесом мыши (Zoom) и ставить курсор.
    """
    selectionChanged = pyqtSignal(int, int)      # (start_frame, end_frame)
    seekRequested = pyqtSignal(int)              # frame
    clipSelected = pyqtSignal(str)               # clip_id
    clipMoved = pyqtSignal(str, int, int)        # (clip_id, new_start_frame, new_track_index)

    HEADER_WIDTH = 135
    MINIMAP_HEIGHT = 32

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(280)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip(
            "Многодорожечный таймлайн:\n"
            "• Клик и перетаскивание клипа — двигать кусок по времени или на другую дорожку\n"
            "• Shift + протягивание мышью — выделить отрезок для изменения громкости или обрезки\n"
            "• Колесо мыши — приблизить/отдалить (Zoom)"
        )

        self._timeline: Optional[NonDestructiveTimeline] = None
        self._audio: np.ndarray = np.zeros((0, 1), dtype=np.float32)
        self._sample_rate: int = 48000
        self.show_spectrogram: bool = False
        self._spec_image: Optional[QImage] = None

        self.view_start_frame: int = 0
        self.view_end_frame: int = 48000 * 10
        self.playhead_frame: int = 0
        self.selection_range: Optional[Tuple[int, int]] = None
        self.selected_clip_id: Optional[str] = None
        self.active_track_index: int = 0

        self.markers: List[TimelineMarker] = []
        self.regions: List[TimelineRegion] = []

        # Состояние перетаскивания (Drag & Drop клипов или выделения)
        self._drag_mode: Optional[str] = None  # "MOVE_CLIP" или "SELECT_RANGE"
        self._drag_clip_id: Optional[str] = None
        self._drag_offset_frames: int = 0
        self._drag_start_frame: int = 0

    def set_timeline_data(self, timeline: NonDestructiveTimeline, mixed_audio: np.ndarray) -> None:
        self._timeline = timeline
        self._audio = mixed_audio if mixed_audio.ndim == 2 else mixed_audio[:, np.newaxis]
        self._sample_rate = max(1, timeline.sample_rate)
        self.markers = timeline.markers
        self.regions = timeline.regions
        total = max(self._sample_rate * 5, timeline.total_frames + self._sample_rate * 2)
        self.view_start_frame = 0
        self.view_end_frame = total
        self._spec_image = None
        if self.show_spectrogram:
            self._rebuild_spectrogram_image()
        self.update()

    def set_audio_data(self, audio: np.ndarray, sample_rate: int,
                       markers: List[TimelineMarker], regions: List[TimelineRegion]) -> None:
        self._audio = audio if audio.ndim == 2 else audio[:, np.newaxis]
        self._sample_rate = max(1, sample_rate)
        self.markers = markers
        self.regions = regions
        self.view_start_frame = 0
        self.view_end_frame = max(self._sample_rate * 5, self._audio.shape[0])
        self._spec_image = None
        if self.show_spectrogram:
            self._rebuild_spectrogram_image()
        self.update()

    def toggle_spectrogram(self, enabled: bool) -> None:
        self.show_spectrogram = enabled
        if enabled and self._spec_image is None:
            self._rebuild_spectrogram_image()
        self.update()

    def _rebuild_spectrogram_image(self) -> None:
        if self._audio.shape[0] < 512:
            self._spec_image = None
            return
        mono = np.mean(self._audio, axis=1)
        _, _, sxx = signal.spectrogram(mono, fs=self._sample_rate, nperseg=512, noverlap=384)
        sxx_db = 10.0 * np.log10(np.flipud(sxx) + 1e-10)
        vmin, vmax = np.percentile(sxx_db, 10), np.percentile(sxx_db, 99)
        norm = np.clip((sxx_db - vmin) / max(1e-5, vmax - vmin), 0.0, 1.0)
        h, w = norm.shape
        rgb = np.zeros((h, w, 4), dtype=np.uint8)
        rgb[..., 0] = (norm ** 0.7 * 255).astype(np.uint8)
        rgb[..., 1] = (norm ** 1.6 * 220).astype(np.uint8)
        rgb[..., 2] = ((1.0 - norm * 0.5) * 140).astype(np.uint8)
        rgb[..., 3] = 255
        self._spec_image = QImage(rgb.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = self.width()
        h = self.height()
        hdr_w = self.HEADER_WIDTH
        mini_h = self.MINIMAP_HEIGHT
        canvas_w = max(1, w - hdr_w)
        main_rect = QRectF(hdr_w, mini_h + 4, canvas_w, h - mini_h - 4)

        painter.fillRect(self.rect(), QColor("#12161F"))
        painter.fillRect(QRectF(0, 0, w, mini_h), QColor("#1A202C"))

        total = max(1, self.view_end_frame, self._audio.shape[0])
        span = max(1, self.view_end_frame - self.view_start_frame)

        # 1. Мини-карта сверху
        painter.setPen(QColor("#90A4AE"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(8, 20, "МИНИ-КАРТА")
        if self._audio.shape[0] > 0:
            self._draw_buffer_envelope(painter, QRectF(hdr_w, 2, canvas_w, mini_h - 4),
                                       self._audio[:, 0], 0, total, QColor("#4FC3F7"))
            vx1 = hdr_w + (self.view_start_frame / total) * canvas_w
            vx2 = hdr_w + (self.view_end_frame / total) * canvas_w
            painter.fillRect(QRectF(vx1, 0, max(6.0, vx2 - vx1), mini_h), QColor(0, 188, 212, 65))

        # 2. Спектрограмма ИЛИ Многодорожечный режим с клипами-объектами
        if self.show_spectrogram and self._spec_image is not None:
            painter.drawImage(main_rect, self._spec_image)
        elif self._timeline is not None:
            n_tracks = max(1, len(self._timeline.tracks))
            track_h = main_rect.height() / n_tracks

            for t_idx, trk in enumerate(self._timeline.tracks):
                ty = main_rect.top() + t_idx * track_h
                t_rect = QRectF(hdr_w, ty, canvas_w, track_h - 2)
                hdr_rect = QRectF(0, ty, hdr_w - 2, track_h - 2)

                # Левая шапка дорожки
                bg_hdr = QColor("#263238") if t_idx == self.active_track_index else QColor("#1C222D")
                painter.fillRect(hdr_rect, bg_hdr)
                painter.setPen(QColor("#00E5FF") if t_idx == self.active_track_index else QColor("#CFD8DC"))
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.drawText(hdr_rect.adjusted(8, 6, -4, -4), Qt.AlignmentFlag.AlignTop, trk.name)

                status_txt = f"{trk.gain_db:+.1f} дБ" + (" [MUTE]" if trk.muted else "")
                painter.setPen(QColor("#FF5252") if trk.muted else QColor("#80CBC4"))
                painter.setFont(QFont("Segoe UI", 8))
                painter.drawText(hdr_rect.adjusted(8, 24, -4, -4), Qt.AlignmentFlag.AlignTop, status_txt)

                # Фон самой дорожки
                painter.fillRect(t_rect, QColor("#161B24") if t_idx % 2 == 0 else QColor("#191F2A"))
                painter.setPen(QPen(QColor("#263238"), 1.0))
                painter.drawLine(int(hdr_w), int(ty + track_h - 1), int(w), int(ty + track_h - 1))

            # Отрисовка клипов-объектов на дорожках
            for clip in self._timeline.clips:
                if clip.track_index >= n_tracks:
                    continue
                x1 = hdr_w + ((clip.timeline_start_frame - self.view_start_frame) / span) * canvas_w
                x2 = hdr_w + ((clip.timeline_end_frame - self.view_start_frame) / span) * canvas_w
                if x2 < hdr_w or x1 > w:
                    continue

                ty = main_rect.top() + clip.track_index * track_h + 3
                c_rect = QRectF(max(hdr_w, x1), ty, min(w, x2) - max(hdr_w, x1), track_h - 8)

                base_color = QColor(clip.color)
                base_color.setAlpha(185)
                painter.setBrush(base_color)
                is_sel = (clip.clip_id == self.selected_clip_id)
                pen_col = QColor("#FFFFFF") if is_sel else QColor("#90CAF9")
                painter.setPen(QPen(pen_col, 2.2 if is_sel else 1.0))
                painter.drawRoundedRect(c_rect, 5.0, 5.0)
                painter.setBrush(Qt.BrushStyle.NoBrush)

                # Форма волны внутри клипа
                src_buf = self._timeline.get_source_preview(clip.source_id)
                if src_buf.size > 0:
                    gain_lin = float(10.0 ** (clip.gain_db / 20.0))
                    sub = src_buf[clip.source_start_frame : clip.source_start_frame + clip.frame_count] * gain_lin
                    self._draw_buffer_envelope(painter, c_rect.adjusted(2, 14, -2, -2),
                                               sub, 0, len(sub), QColor("#E1F5FE"))

                # Заголовок клипа и его громкость
                painter.setPen(QColor("#FFFFFF"))
                painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                label = f"{clip.name} ({clip.gain_db:+.1f} дБ)"
                painter.drawText(c_rect.adjusted(6, 2, -4, -2), Qt.AlignmentFlag.AlignTop, label)

        # 3. Выделенный диапазон времени (Желтая зона)
        if self.selection_range is not None:
            s, e = self.selection_range
            sx1 = hdr_w + ((s - self.view_start_frame) / span) * canvas_w
            sx2 = hdr_w + ((e - self.view_start_frame) / span) * canvas_w
            painter.fillRect(QRectF(max(hdr_w, sx1), main_rect.top(), sx2 - sx1, main_rect.height()),
                             QColor(255, 213, 79, 65))

        # 4. Маркеры и красный курсор воспроизведения
        for m in self.markers:
            mx = hdr_w + ((m.frame - self.view_start_frame) / span) * canvas_w
            if hdr_w <= mx <= w:
                painter.setPen(QPen(QColor(m.color), 1.5, Qt.PenStyle.DashLine))
                painter.drawLine(int(mx), int(main_rect.top()), int(mx), int(main_rect.bottom()))
                painter.drawText(int(mx) + 4, int(main_rect.top()) + 14, m.name)

        px = hdr_w + ((self.playhead_frame - self.view_start_frame) / span) * canvas_w
        if hdr_w <= px <= w:
            painter.setPen(QPen(QColor("#FF5252"), 2.0))
            painter.drawLine(int(px), 0, int(px), h)

    @staticmethod
    def _draw_buffer_envelope(painter: QPainter, rect: QRectF, mono_buf: np.ndarray,
                              start_frame: int, end_frame: int, color: QColor) -> None:
        segment = mono_buf[max(0, start_frame) : min(mono_buf.shape[0], end_frame)]
        if segment.size == 0 or rect.width() <= 2:
            return
        width_px = max(1, int(rect.width()))
        step = max(1, segment.size // width_px)
        n_bins = segment.size // step
        if n_bins == 0:
            return
        reshaped = np.clip(segment[: n_bins * step].reshape(n_bins, step), -1.0, 1.0)
        mins = np.min(reshaped, axis=1)
        maxs = np.max(reshaped, axis=1)

        mid_y = rect.center().y()
        half_h = rect.height() * 0.45
        painter.setPen(QPen(color, 1.0))
        for x in range(n_bins):
            y1 = mid_y - maxs[x] * half_h
            y2 = mid_y - mins[x] * half_h
            painter.drawLine(int(rect.left() + x), int(y1), int(rect.left() + x), int(y2))

    # ------------------------------------------------------------------
    # Интерактивность: клик, перетаскивание клипов, выделение отрезка, Zoom
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:
        total = max(self._sample_rate * 5, self._timeline.total_frames + self._sample_rate * 5 if self._timeline else 48000)
        factor = 0.8 if event.angleDelta().y() > 0 else 1.25
        span = max(512, int((self.view_end_frame - self.view_start_frame) * factor))
        span = min(total, span)
        canvas_w = max(1, self.width() - self.HEADER_WIDTH)
        center_ratio = np.clip((event.position().x() - self.HEADER_WIDTH) / canvas_w, 0.0, 1.0)
        anchor = self.view_start_frame + int((self.view_end_frame - self.view_start_frame) * center_ratio)
        self.view_start_frame = max(0, min(total - span, anchor - int(span * center_ratio)))
        self.view_end_frame = self.view_start_frame + span
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        t_idx = self._y_to_track_index(pos.y())
        self.active_track_index = t_idx

        # Клик по левой шапке дорожки просто выбирает активную дорожку для записи
        if pos.x() < self.HEADER_WIDTH:
            self.update()
            return

        frame = self._x_to_frame(pos.x())
        self.playhead_frame = frame
        self.seekRequested.emit(frame)

        # Проверяем, попали ли мы в конкретный клип-объект (если не зажат Shift)
        hit_clip = None if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) else self._find_clip_at(frame, t_idx)

        if hit_clip is not None and event.button() == Qt.MouseButton.LeftButton:
            self.selected_clip_id = hit_clip.clip_id
            self.clipSelected.emit(hit_clip.clip_id)
            self._drag_mode = "MOVE_CLIP"
            self._drag_clip_id = hit_clip.clip_id
            self._drag_offset_frames = frame - hit_clip.timeline_start_frame
        else:
            self._drag_mode = "SELECT_RANGE"
            self._drag_start_frame = frame
            self.selection_range = None
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        curr_frame = self._x_to_frame(pos.x())
        curr_track = self._y_to_track_index(pos.y())

        if self._drag_mode == "MOVE_CLIP" and self._timeline and self._drag_clip_id:
            for c in self._timeline.clips:
                if c.clip_id == self._drag_clip_id:
                    c.timeline_start_frame = max(0, curr_frame - self._drag_offset_frames)
                    c.track_index = max(0, min(len(self._timeline.tracks) - 1, curr_track))
                    break
            self.update()
        elif self._drag_mode == "SELECT_RANGE":
            s, e = min(self._drag_start_frame, curr_frame), max(self._drag_start_frame, curr_frame)
            if e - s > 64:
                self.selection_range = (s, e)
                self.selectionChanged.emit(s, e)
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_mode == "MOVE_CLIP" and self._timeline and self._drag_clip_id:
            for c in self._timeline.clips:
                if c.clip_id == self._drag_clip_id:
                    self.clipMoved.emit(c.clip_id, c.timeline_start_frame, c.track_index)
                    break
        self._drag_mode = None
        self._drag_clip_id = None

    def _find_clip_at(self, frame: int, track_index: int) -> Optional[ClipSlice]:
        if not self._timeline:
            return None
        for c in reversed(self._timeline.clips):
            if c.track_index == track_index and c.timeline_start_frame <= frame <= c.timeline_end_frame:
                return c
        return None

    def _x_to_frame(self, x: float) -> int:
        canvas_w = max(1, self.width() - self.HEADER_WIDTH)
        ratio = np.clip((x - self.HEADER_WIDTH) / canvas_w, 0.0, 1.0)
        span = self.view_end_frame - self.view_start_frame
        return int(self.view_start_frame + ratio * span)

    def _y_to_track_index(self, y: float) -> int:
        if not self._timeline or not self._timeline.tracks:
            return 0
        main_top = self.MINIMAP_HEIGHT + 4
        main_h = max(1, self.height() - main_top)
        n_tracks = len(self._timeline.tracks)
        idx = int((y - main_top) / (main_h / n_tracks))
        return max(0, min(n_tracks - 1, idx))


class LevelAndPhaseMeterWidget(QWidget):
    """Индикатор уровня (Peak + RMS в дБ) и стерео-фазовый коррелятор (-1 .. +1)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(185, 68)
        self.setToolTip("Индикатор пиковой/средней громкости (Peak/RMS) и фазовой совместимости каналов.")
        self.peak_db: float = -96.0
        self.rms_db: float = -96.0
        self.phase: float = 1.0

    def update_meters(self, peak_db: float, rms_db: float, phase: float) -> None:
        self.peak_db = peak_db
        self.rms_db = rms_db
        self.phase = phase
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#181C24"))

        def db_to_norm(db: float) -> float:
            return float(np.clip((db + 60.0) / 60.0, 0.0, 1.0))

        rms_w = int(db_to_norm(self.rms_db) * (w - 16))
        peak_w = int(db_to_norm(self.peak_db) * (w - 16))

        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor("#B0BEC5"))
        p.drawText(8, 14, f"Peak: {self.peak_db:5.1f} dB | RMS: {self.rms_db:5.1f} dB")

        p.fillRect(8, 20, w - 16, 14, QColor("#263238"))
        bar_color = QColor("#FF5252") if self.peak_db > -1.0 else QColor("#00E676")
        p.fillRect(8, 20, rms_w, 14, QColor("#00B0FF"))
        p.setPen(QPen(bar_color, 2.5))
        p.drawLine(8 + peak_w, 19, 8 + peak_w, 35)

        p.setPen(QColor("#B0BEC5"))
        p.drawText(8, 50, f"Фаза: {self.phase:+.2f}")
        p.fillRect(8, 54, w - 16, 10, QColor("#263238"))
        center_x = w // 2
        phase_x = int(center_x + self.phase * ((w - 16) / 2.0))
        p.setPen(QPen(QColor("#FFC107"), 3.0))
        p.drawLine(center_x, 54, phase_x, 54)