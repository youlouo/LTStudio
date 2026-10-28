# ui/widgets.py
from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy import signal
from PyQt6.QtCore import Qt, QRectF, QPoint, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QPolygonF, QFont, QImage, QMouseEvent, QWheelEvent, QKeyEvent
from PyQt6.QtWidgets import QWidget

from project.edl import NonDestructiveTimeline, ClipSlice, TimelineMarker, TimelineRegion


class WaveformSpectrogramWidget(QWidget):
    """
    Многодорожечный рабочий холст LTStudio:
    - Отрисовывает в реальном времени бегущий клип во время записи с микрофона.
    - Отображает отметки маркеров на верхней полосе «ОБЗОР ПРОЕКТА» и позволяет перетаскивать/удалять маркеры мышью.
    - Обрабатывает клавиши стрелок (Влево, Вправо, Вверх, Вниз) для перемещения выделенных клипов.
    """
    selectionChanged = pyqtSignal(int, int)                             # (start_frame, end_frame)
    clipsSelectionChanged = pyqtSignal(list)                            # list[clip_id]
    clipsMoved = pyqtSignal()                                           # групповое перемещение мышью
    clipsNudgedByKeys = pyqtSignal(int, int)                            # (delta_frames, delta_track)
    markerMoved = pyqtSignal(str, int)                                  # (marker_id, new_frame)
    contextMenuRequested = pyqtSignal(QPoint, str, str, int, int, bool) # (pos, clip_id, marker_id, track_idx, frame, is_range)

    HEADER_WIDTH = 155
    MINIMAP_HEIGHT = 32

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setToolTip(
            "Многодорожечный таймлайн:\n"
            "- Верхняя полоса «ОБЗОР ПРОЕКТА»: навигация по проекту + отметки всех маркеров\n"
            "- Клавиши Стрелок (Влево/Вправо/Вверх/Вниз): перемещение выделенных клипов по времени и дорожкам\n"
            "- Маркеры (желтые линии): потяните левой кнопкой мыши для перемещения или нажмите ПКМ для удаления"
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

        self.selected_clip_ids: Set[str] = set()
        self.active_track_index: int = 0

        self.markers: List[TimelineMarker] = []
        self.regions: List[TimelineRegion] = []

        # Живая отрисовка бегущего клипа во время записи
        self.live_rec_active: bool = False
        self.live_rec_track: int = 0
        self.live_rec_start_frame: int = 0
        self.live_rec_mono: np.ndarray = np.zeros(0, dtype=np.float32)

        # Режимы перетаскивания: "PAN_MINIMAP", "MOVE_CLIPS", "MOVE_MARKER", "SELECT_RANGE"
        self._drag_mode: Optional[str] = None
        self._drag_marker_id: Optional[str] = None
        self._drag_anchor_frame: int = 0
        self._drag_anchor_track: int = 0
        self._initial_clip_positions: Dict[str, Tuple[int, int]] = {}
        self._drag_start_frame: int = 0

    @property
    def selected_clip_id(self) -> Optional[str]:
        return next(iter(self.selected_clip_ids), None)

    @selected_clip_id.setter
    def selected_clip_id(self, val: Optional[str]) -> None:
        self.selected_clip_ids.clear()
        if val:
            self.selected_clip_ids.add(val)

    def set_live_recording_state(self, active: bool, track_idx: int = 0,
                                 start_frame: int = 0, live_mono: Optional[np.ndarray] = None) -> None:
        self.live_rec_active = active
        self.live_rec_track = track_idx
        self.live_rec_start_frame = start_frame
        self.live_rec_mono = live_mono if live_mono is not None else np.zeros(0, dtype=np.float32)
        if active:
            live_end = start_frame + len(self.live_rec_mono)
            if live_end > self.view_end_frame - int(0.5 * self._sample_rate):
                span = max(self._sample_rate * 5, self.view_end_frame - self.view_start_frame)
                self.view_end_frame = live_end + int(2.0 * self._sample_rate)
                self.view_start_frame = max(0, self.view_end_frame - span)
        self.update()

    def _total_project_frames(self) -> int:
        tl_frames = self._timeline.total_frames if self._timeline else self._audio.shape[0]
        if self.live_rec_active:
            tl_frames = max(tl_frames, self.live_rec_start_frame + len(self.live_rec_mono))
        return max(self._sample_rate * 6, int(tl_frames * 1.15) + self._sample_rate * 2)

    def set_timeline_data(self, timeline: NonDestructiveTimeline, mixed_audio: np.ndarray) -> None:
        self._timeline = timeline
        self._audio = mixed_audio if mixed_audio.ndim == 2 else mixed_audio[:, np.newaxis]
        self._sample_rate = max(1, timeline.sample_rate)
        self.markers = timeline.markers
        self.regions = timeline.regions
        total = self._total_project_frames()
        if self.view_end_frame <= self.view_start_frame or self.view_end_frame > total * 2:
            self.view_start_frame = 0
            self.view_end_frame = total
        else:
            self.view_end_frame = max(self.view_start_frame + 1024, min(total, self.view_end_frame))
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
        main_rect = QRectF(hdr_w, mini_h + 2, canvas_w, h - mini_h - 2)

        painter.fillRect(self.rect(), QColor("#11151C"))
        painter.fillRect(QRectF(0, 0, w, mini_h), QColor("#181E29"))

        total = max(1, self._total_project_frames(), self.view_end_frame)
        span = max(1, self.view_end_frame - self.view_start_frame)

        # 1. Верхняя полоса «ОБЗОР ПРОЕКТА» с волной, маркерами и голубой рамкой навигации
        painter.setPen(QColor("#90CAF9"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(8, 20, "ОБЗОР ПРОЕКТА")
        if self._audio.shape[0] > 0:
            self._draw_buffer_envelope(painter, QRectF(hdr_w, 2, canvas_w, mini_h - 4),
                                       self._audio[:, 0], 0, total, QColor("#4FC3F7"))

        # Отметки маркеров прямо на полосе «ОБЗОР ПРОЕКТА»
        for m in self.markers:
            mini_mx = hdr_w + (m.frame / total) * canvas_w
            if hdr_w <= mini_mx <= w:
                painter.setPen(QPen(QColor(m.color), 1.5))
                painter.drawLine(int(mini_mx), 1, int(mini_mx), mini_h - 1)
                flag_poly = QPolygonF([
                    QPointF(mini_mx, 2.0),
                    QPointF(mini_mx + 6.0, 5.0),
                    QPointF(mini_mx, 8.0),
                ])
                painter.setBrush(QColor(m.color))
                painter.drawPolygon(flag_poly)
                painter.setBrush(Qt.BrushStyle.NoBrush)

        vx1 = hdr_w + (self.view_start_frame / total) * canvas_w
        vx2 = hdr_w + (self.view_end_frame / total) * canvas_w
        zone_rect = QRectF(vx1, 1, max(12.0, min(w - vx1, vx2 - vx1)), mini_h - 2)
        painter.fillRect(zone_rect, QColor(0, 188, 212, 70))
        painter.setPen(QPen(QColor("#00E5FF"), 1.5))
        painter.drawRect(zone_rect)

        # 2. Спектрограмма или Многодорожечный режим
        if self.show_spectrogram and self._spec_image is not None:
            painter.drawImage(main_rect, self._spec_image)
        elif self._timeline is not None:
            n_tracks = max(1, len(self._timeline.tracks))
            track_h = main_rect.height() / n_tracks

            for t_idx, trk in enumerate(self._timeline.tracks):
                ty = main_rect.top() + t_idx * track_h
                t_rect = QRectF(hdr_w, ty, canvas_w, track_h - 1)
                hdr_rect = QRectF(0, ty, hdr_w - 2, track_h - 1)

                bg_hdr = QColor("#232D3B") if t_idx == self.active_track_index else QColor("#181F29")
                painter.fillRect(hdr_rect, bg_hdr)
                painter.fillRect(QRectF(0, ty, 5, track_h - 1), QColor(trk.color))

                painter.setPen(QColor("#ECEFF1") if t_idx == self.active_track_index else QColor("#B0BEC5"))
                painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                painter.drawText(hdr_rect.adjusted(10, 6, -4, -4), Qt.AlignmentFlag.AlignTop, trk.name)

                status_txt = f"Уровень: {trk.gain_db:+.1f} дБ" + (" [БЕЗ ЗВУКА]" if trk.muted else "")
                painter.setPen(QColor("#EF5350") if trk.muted else QColor("#78909C"))
                painter.setFont(QFont("Segoe UI", 8))
                painter.drawText(hdr_rect.adjusted(10, 22, -4, -4), Qt.AlignmentFlag.AlignTop, status_txt)

                painter.fillRect(t_rect, QColor("#141922") if t_idx % 2 == 0 else QColor("#171D27"))
                painter.setPen(QPen(QColor("#222B38"), 1.0))
                painter.drawLine(int(hdr_w), int(ty + track_h - 1), int(w), int(ty + track_h - 1))

            # Отрисовка существующих клипов
            for clip in self._timeline.clips:
                if clip.track_index >= n_tracks:
                    continue
                x1 = hdr_w + ((clip.timeline_start_frame - self.view_start_frame) / span) * canvas_w
                x2 = hdr_w + ((clip.timeline_end_frame - self.view_start_frame) / span) * canvas_w
                if x2 < hdr_w or x1 > w:
                    continue

                ty = main_rect.top() + clip.track_index * track_h + 3
                c_rect = QRectF(max(hdr_w, x1), ty, max(4.0, min(w, x2) - max(hdr_w, x1)), track_h - 7)

                base_color = QColor(clip.color)
                base_color.setAlpha(180)
                painter.setBrush(base_color)
                is_sel = (clip.clip_id in self.selected_clip_ids)
                pen_col = QColor("#FFEB3B") if is_sel else QColor("#90CAF9")
                painter.setPen(QPen(pen_col, 2.4 if is_sel else 1.0))
                painter.drawRoundedRect(c_rect, 3.0, 3.0)
                painter.setBrush(Qt.BrushStyle.NoBrush)

                src_buf = self._timeline.get_source_preview(clip.source_id)
                if src_buf.size > 0:
                    gain_lin = float(10.0 ** (clip.gain_db / 20.0))
                    sub = src_buf[clip.source_start_frame : clip.source_start_frame + clip.frame_count] * gain_lin
                    self._draw_buffer_envelope(painter, c_rect.adjusted(2, 14, -2, -2),
                                               sub, 0, len(sub), QColor("#E1F5FE"))

                if clip.fade_in_frames > 0 and clip.effective_frame_count > 0:
                    fi_w = (clip.fade_in_frames / span) * canvas_w
                    fi_x = min(c_rect.right(), x1 + fi_w)
                    if fi_x > c_rect.left():
                        poly = QPolygonF([
                            QPointF(c_rect.left(), c_rect.top()),
                            QPointF(fi_x, c_rect.top()),
                            QPointF(c_rect.left(), c_rect.bottom()),
                        ])
                        painter.setBrush(QColor(0, 0, 0, 115))
                        painter.setPen(QPen(QColor("#FFCA28"), 1.5))
                        painter.drawPolygon(poly)
                        painter.setBrush(Qt.BrushStyle.NoBrush)

                if clip.fade_out_frames > 0 and clip.effective_frame_count > 0:
                    fo_w = (clip.fade_out_frames / span) * canvas_w
                    fo_x = max(c_rect.left(), x2 - fo_w)
                    if fo_x < c_rect.right():
                        poly = QPolygonF([
                            QPointF(fo_x, c_rect.top()),
                            QPointF(c_rect.right(), c_rect.top()),
                            QPointF(c_rect.right(), c_rect.bottom()),
                        ])
                        painter.setBrush(QColor(0, 0, 0, 115))
                        painter.setPen(QPen(QColor("#FFCA28"), 1.5))
                        painter.drawPolygon(poly)
                        painter.setBrush(Qt.BrushStyle.NoBrush)

                painter.setPen(QColor("#FFFFFF"))
                painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                badges = f" ({clip.gain_db:+.1f} дБ)"
                if abs(clip.speed_rate - 1.0) > 1e-2:
                    badges += f" [{clip.speed_rate:.2g}x]"
                if clip.dsp_profile_name:
                    badges += f" [Профиль: {clip.dsp_profile_name}]"
                painter.drawText(c_rect.adjusted(6, 2, -4, -2), Qt.AlignmentFlag.AlignTop, clip.name + badges)

            # 2Б. Отрисовка ЖИВОГО БЕГУЩЕГО КЛИПА во время записи
            if self.live_rec_active and 0 <= self.live_rec_track < n_tracks:
                rec_len = len(self.live_rec_mono)
                rx1 = hdr_w + ((self.live_rec_start_frame - self.view_start_frame) / span) * canvas_w
                rx2 = hdr_w + ((self.live_rec_start_frame + max(1024, rec_len) - self.view_start_frame) / span) * canvas_w
                if rx2 >= hdr_w and rx1 <= w:
                    r_ty = main_rect.top() + self.live_rec_track * track_h + 3
                    r_rect = QRectF(max(hdr_w, rx1), r_ty, max(6.0, min(w, rx2) - max(hdr_w, rx1)), track_h - 7)

                    rec_bg = QColor("#B71C1C")
                    rec_bg.setAlpha(190)
                    painter.setBrush(rec_bg)
                    painter.setPen(QPen(QColor("#FF5252"), 2.2))
                    painter.drawRoundedRect(r_rect, 3.0, 3.0)
                    painter.setBrush(Qt.BrushStyle.NoBrush)

                    if rec_len > 0:
                        self._draw_buffer_envelope(
                            painter, r_rect.adjusted(2, 14, -2, -2),
                            self.live_rec_mono, 0, rec_len, QColor("#FFCDD2")
                        )
                    rec_sec = rec_len / max(1, self._sample_rate)
                    painter.setPen(QColor("#FFFFFF"))
                    painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                    painter.drawText(r_rect.adjusted(6, 2, -4, -2), Qt.AlignmentFlag.AlignTop, f"ЗАПИСЬ... ({rec_sec:.1f} с)")

        # 3. Оранжевая зона выделенного диапазона времени
        if self.selection_range is not None:
            s, e = self.selection_range
            sx1 = hdr_w + ((s - self.view_start_frame) / span) * canvas_w
            sx2 = hdr_w + ((e - self.view_start_frame) / span) * canvas_w
            sel_rect = QRectF(max(hdr_w, sx1), main_rect.top(), max(2.0, sx2 - max(hdr_w, sx1)), main_rect.height())
            painter.fillRect(sel_rect, QColor(255, 167, 38, 70))
            painter.setPen(QPen(QColor("#FFA726"), 1.5))
            painter.drawRect(sel_rect)

        # 4. Маркеры на основном таймлайне (с флажками для удобного перетаскивания) и красный курсор
        for m in self.markers:
            mx = hdr_w + ((m.frame - self.view_start_frame) / span) * canvas_w
            if hdr_w <= mx <= w:
                painter.setPen(QPen(QColor(m.color), 1.8, Qt.PenStyle.DashLine))
                painter.drawLine(int(mx), int(main_rect.top()), int(mx), int(main_rect.bottom()))
                lbl_rect = QRectF(mx, main_rect.top(), 68, 16)
                painter.fillRect(lbl_rect, QColor(m.color))
                painter.setPen(QColor("#11151C"))
                painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                painter.drawText(lbl_rect.adjusted(4, 1, -2, -1), Qt.AlignmentFlag.AlignVCenter, m.name)

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
    # Клавиши Стрелок (Влево, Вправо, Вверх, Вниз) и Мышь (Перетаскивание маркеров)
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self.selected_clip_ids:
            step_sec = 0.5 if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) else 0.1
            step_frames = max(1, int(round(step_sec * self._sample_rate)))
            if event.key() == Qt.Key.Key_Left:
                self.clipsNudgedByKeys.emit(-step_frames, 0)
                event.accept()
                return
            if event.key() == Qt.Key.Key_Right:
                self.clipsNudgedByKeys.emit(step_frames, 0)
                event.accept()
                return
            if event.key() == Qt.Key.Key_Up:
                self.clipsNudgedByKeys.emit(0, -1)
                event.accept()
                return
            if event.key() == Qt.Key.Key_Down:
                self.clipsNudgedByKeys.emit(0, 1)
                event.accept()
                return
        super().keyPressEvent(event)

    def _find_marker_near_x(self, x_px: float, tolerance_px: float = 9.0) -> Optional[TimelineMarker]:
        if x_px < self.HEADER_WIDTH:
            return None
        canvas_w = max(1, self.width() - self.HEADER_WIDTH)
        span = max(1, self.view_end_frame - self.view_start_frame)
        for m in self.markers:
            mx = self.HEADER_WIDTH + ((m.frame - self.view_start_frame) / span) * canvas_w
            if abs(x_px - mx) <= tolerance_px or (mx <= x_px <= mx + 64):
                return m
        return None

    def _pan_minimap_to_x(self, x: float) -> None:
        canvas_w = max(1, self.width() - self.HEADER_WIDTH)
        ratio = float(np.clip((x - self.HEADER_WIDTH) / canvas_w, 0.0, 1.0))
        total = self._total_project_frames()
        span = max(512, self.view_end_frame - self.view_start_frame)
        span = min(total, span)
        center_frame = int(ratio * total)
        new_start = max(0, min(total - span, center_frame - span // 2))
        self.view_start_frame = new_start
        self.view_end_frame = new_start + span
        self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        total = self._total_project_frames()
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
        self.setFocus()
        pos = event.position()

        # 1. Навигация по полосе «ОБЗОР ПРОЕКТА»
        if pos.y() <= self.MINIMAP_HEIGHT and pos.x() >= self.HEADER_WIDTH:
            if event.button() == Qt.MouseButton.LeftButton:
                self._drag_mode = "PAN_MINIMAP"
                self._pan_minimap_to_x(pos.x())
            return

        t_idx = self._y_to_track_index(pos.y())
        self.active_track_index = t_idx

        if pos.x() < self.HEADER_WIDTH:
            if event.button() == Qt.MouseButton.RightButton:
                self.contextMenuRequested.emit(event.globalPosition().toPoint(), "", "", t_idx, self.playhead_frame, False)
            self.update()
            return

        frame = self._x_to_frame(pos.x())
        hit_marker = self._find_marker_near_x(pos.x()) if pos.y() <= self.MINIMAP_HEIGHT + 24 else self._find_marker_near_x(pos.x(), 6.0)
        hit_clip = self._find_clip_at(frame, t_idx)

        # 2. Правый клик: передаем также ID маркера, если кликнули по маркеру
        if event.button() == Qt.MouseButton.RightButton:
            in_orange_range = (
                self.selection_range is not None
                and self.selection_range[0] <= frame <= self.selection_range[1]
            )
            cid = hit_clip.clip_id if hit_clip else ""
            mid = hit_marker.marker_id if hit_marker else ""
            if hit_clip and hit_clip.clip_id not in self.selected_clip_ids:
                self.selected_clip_ids = {hit_clip.clip_id}
                self.clipsSelectionChanged.emit(list(self.selected_clip_ids))
            self.update()
            self.contextMenuRequested.emit(event.globalPosition().toPoint(), cid, mid, t_idx, frame, in_orange_range)
            return

        # 3. Левый клик: перетаскивание маркера ИЛИ клипов ИЛИ выделение зоны
        if event.button() == Qt.MouseButton.LeftButton:
            if hit_marker is not None and (pos.y() <= self.MINIMAP_HEIGHT + 22 or hit_clip is None):
                self._drag_mode = "MOVE_MARKER"
                self._drag_marker_id = hit_marker.marker_id
                return

            ctrl_pressed = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            shift_pressed = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

            if shift_pressed or hit_clip is None:
                self._drag_mode = "SELECT_RANGE"
                self._drag_start_frame = frame
                self.selection_range = None
                if hit_clip is None and not ctrl_pressed:
                    self.selected_clip_ids.clear()
                    self.clipsSelectionChanged.emit([])
            else:
                if ctrl_pressed:
                    if hit_clip.clip_id in self.selected_clip_ids:
                        self.selected_clip_ids.remove(hit_clip.clip_id)
                    else:
                        self.selected_clip_ids.add(hit_clip.clip_id)
                else:
                    if hit_clip.clip_id not in self.selected_clip_ids:
                        self.selected_clip_ids = {hit_clip.clip_id}

                self.clipsSelectionChanged.emit(list(self.selected_clip_ids))
                self._drag_mode = "MOVE_CLIPS"
                self._drag_anchor_frame = frame
                self._drag_anchor_track = t_idx
                self._initial_clip_positions = {
                    c.clip_id: (c.timeline_start_frame, c.track_index)
                    for c in (self._timeline.clips if self._timeline else [])
                    if c.clip_id in self.selected_clip_ids
                }
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()

        if self._drag_mode == "PAN_MINIMAP":
            self._pan_minimap_to_x(pos.x())
            return

        curr_frame = self._x_to_frame(pos.x())
        curr_track = self._y_to_track_index(pos.y())

        if self._drag_mode == "MOVE_MARKER" and self._drag_marker_id:
            for m in self.markers:
                if m.marker_id == self._drag_marker_id:
                    m.frame = max(0, curr_frame)
                    break
            self.update()
            return

        if self._drag_mode == "MOVE_CLIPS" and self._timeline and self._initial_clip_positions:
            delta_frames = curr_frame - self._drag_anchor_frame
            delta_track = curr_track - self._drag_anchor_track
            n_tracks = len(self._timeline.tracks)
            for c in self._timeline.clips:
                if c.clip_id in self._initial_clip_positions:
                    orig_start, orig_trk = self._initial_clip_positions[c.clip_id]
                    c.timeline_start_frame = max(0, orig_start + delta_frames)
                    c.track_index = max(0, min(n_tracks - 1, orig_trk + delta_track))
            self.update()
        elif self._drag_mode == "SELECT_RANGE":
            s, e = min(self._drag_start_frame, curr_frame), max(self._drag_start_frame, curr_frame)
            if e - s > 64:
                self.selection_range = (s, e)
                self.selectionChanged.emit(s, e)
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_mode == "MOVE_MARKER" and self._drag_marker_id:
            for m in self.markers:
                if m.marker_id == self._drag_marker_id:
                    self.markerMoved.emit(m.marker_id, m.frame)
                    break
        elif self._drag_mode == "MOVE_CLIPS" and self._initial_clip_positions:
            self.clipsMoved.emit()
        self._drag_mode = None
        self._drag_marker_id = None
        self._initial_clip_positions.clear()

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
        main_top = self.MINIMAP_HEIGHT + 2
        main_h = max(1, self.height() - main_top)
        n_tracks = len(self._timeline.tracks)
        idx = int((y - main_top) / (main_h / n_tracks))
        return max(0, min(n_tracks - 1, idx))


class LevelAndPhaseMeterWidget(QWidget):
    """Профессиональный измеритель уровня (Peak + RMS в дБ) и фазовый коррелятор (-1 .. +1)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(175, 50)
        self.setMaximumHeight(56)
        self.setToolTip(
            "Измеритель выходного уровня сигнала:\n"
            "- Пик: максимальный мгновенный уровень (рекомендуется не выше -1.0 дБ)\n"
            "- RMS: средняя энергетическая громкость речи\n"
            "- Фаза: моно-совместимость стереоканалов"
        )
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
        p.fillRect(self.rect(), QColor("#141820"))

        def db_to_norm(db: float) -> float:
            return float(np.clip((db + 60.0) / 60.0, 0.0, 1.0))

        rms_w = int(db_to_norm(self.rms_db) * (w - 16))
        peak_w = int(db_to_norm(self.peak_db) * (w - 16))

        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor("#B0BEC5"))
        p.drawText(8, 12, f"Пик: {self.peak_db:5.1f} дБ | RMS: {self.rms_db:5.1f} дБ")

        p.fillRect(8, 16, w - 16, 10, QColor("#263238"))
        bar_color = QColor("#EF5350") if self.peak_db > -1.0 else QColor("#26A69A")
        p.fillRect(8, 16, rms_w, 10, QColor("#0288D1"))
        p.setPen(QPen(bar_color, 2.0))
        p.drawLine(8 + peak_w, 15, 8 + peak_w, 27)

        p.setPen(QColor("#90A4AE"))
        p.drawText(8, 39, f"Фаза: {self.phase:+.2f}")
        p.fillRect(8, 42, w - 16, 5, QColor("#263238"))
        center_x = w // 2
        phase_x = int(center_x + self.phase * ((w - 16) / 2.0))
        p.setPen(QPen(QColor("#FFA000"), 2.5))
        p.drawLine(center_x, 42, phase_x, 42)