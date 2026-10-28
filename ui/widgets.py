# ui/widgets.py
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from scipy import signal
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QImage, QMouseEvent, QWheelEvent
from PyQt6.QtWidgets import QWidget

from project.edl import TimelineMarker, TimelineRegion


class WaveformSpectrogramWidget(QWidget):
    """
    Высокопроизводительный виджет визуализации Waveform + Мини-карта + FFT Спектрограмма.
    Поддерживает Zoom, Pan, выделение фрагмента для зацикливания/редактирования и отображение маркеров.
    """
    selectionChanged = pyqtSignal(int, int)   # (start_frame, end_frame)
    seekRequested = pyqtSignal(int)           # frame

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(240)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Аудио таймлайн и спектрограмма")
        self.setAccessibleDescription(
            "Отображает форму волны или спектрограмму. Используйте колесо мыши для масштабирования и стрелки для навигации."
        )

        self._audio: np.ndarray = np.zeros((0, 1), dtype=np.float32)
        self._sample_rate: int = 48000
        self.show_spectrogram: bool = False
        self._spec_image: Optional[QImage] = None

        self.view_start_frame: int = 0
        self.view_end_frame: int = 48000
        self.playhead_frame: int = 0
        self.selection_range: Optional[Tuple[int, int]] = None
        self.markers: List[TimelineMarker] = []
        self.regions: List[TimelineRegion] = []
        self._drag_start_frame: Optional[int] = None

    def set_audio_data(self, audio: np.ndarray, sample_rate: int,
                       markers: List[TimelineMarker], regions: List[TimelineRegion]) -> None:
        self._audio = audio if audio.ndim == 2 else audio[:, np.newaxis]
        self._sample_rate = max(1, sample_rate)
        self.markers = markers
        self.regions = regions
        self.view_start_frame = 0
        self.view_end_frame = max(1024, self._audio.shape[0])
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
        # Цветовая карта Inferno-like для четкого чтения формант голоса
        rgb[..., 0] = (norm ** 0.7 * 255).astype(np.uint8)       # R
        rgb[..., 1] = (norm ** 1.6 * 220).astype(np.uint8)       # G
        rgb[..., 2] = ((1.0 - norm * 0.5) * 140).astype(np.uint8)  # B
        rgb[..., 3] = 255
        self._spec_image = QImage(rgb.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = self.width()
        h = self.height()
        minimap_h = 34
        main_rect = QRectF(0, minimap_h + 4, w, h - minimap_h - 4)

        # Фон
        painter.fillRect(self.rect(), QColor("#141820"))
        painter.fillRect(QRectF(0, 0, w, minimap_h), QColor("#1C222D"))

        total = max(1, self._audio.shape[0])

        # 1. Отрисовка мини-карты (Minimap)
        if self._audio.shape[0] > 0:
            self._draw_waveform_envelope(painter, QRectF(0, 2, w, minimap_h - 4), 0, total, QColor("#4A6984"))
            vx1 = (self.view_start_frame / total) * w
            vx2 = (self.view_end_frame / total) * w
            painter.fillRect(QRectF(vx1, 0, max(4.0, vx2 - vx1), minimap_h), QColor(0, 188, 212, 65))
            painter.setPen(QPen(QColor("#00BCD4"), 1.5))
            painter.drawRect(QRectF(vx1, 1, max(4.0, vx2 - vx1), minimap_h - 2))

        # 2. Основная область (Spectrogram или Waveform)
        if self.show_spectrogram and self._spec_image is not None:
            painter.drawImage(main_rect, self._spec_image)
        elif self._audio.shape[0] > 0:
            self._draw_waveform_envelope(
                painter, main_rect, self.view_start_frame, self.view_end_frame, QColor("#29B6F6")
            )

        # 3. Выделенный диапазон (Loop / Selection)
        span = max(1, self.view_end_frame - self.view_start_frame)
        if self.selection_range is not None:
            s, e = self.selection_range
            sx1 = ((s - self.view_start_frame) / span) * w
            sx2 = ((e - self.view_start_frame) / span) * w
            painter.fillRect(QRectF(sx1, main_rect.top(), sx2 - sx1, main_rect.height()), QColor(255, 193, 7, 55))

        # 4. Маркеры и курсор воспроизведения
        for m in self.markers:
            mx = ((m.frame - self.view_start_frame) / span) * w
            if 0 <= mx <= w:
                painter.setPen(QPen(QColor(m.color), 1.5, Qt.PenStyle.DashLine))
                painter.drawLine(int(mx), int(main_rect.top()), int(mx), int(main_rect.bottom()))
                painter.drawText(int(mx) + 4, int(main_rect.top()) + 14, m.name)

        px = ((self.playhead_frame - self.view_start_frame) / span) * w
        if 0 <= px <= w:
            painter.setPen(QPen(QColor("#FF5252"), 2.0))
            painter.drawLine(int(px), 0, int(px), h)

    def _draw_waveform_envelope(self, painter: QPainter, rect: QRectF,
                                start_frame: int, end_frame: int, color: QColor) -> None:
        segment = self._audio[max(0, start_frame) : min(self._audio.shape[0], end_frame), 0]
        if segment.size == 0:
            return
        width_px = max(1, int(rect.width()))
        step = max(1, segment.size // width_px)
        n_bins = segment.size // step
        if n_bins == 0:
            return
        reshaped = segment[: n_bins * step].reshape(n_bins, step)
        mins = np.min(reshaped, axis=1)
        maxs = np.max(reshaped, axis=1)

        mid_y = rect.center().y()
        half_h = rect.height() * 0.46
        painter.setPen(QPen(color, 1.0))
        for x in range(n_bins):
            y1 = mid_y - maxs[x] * half_h
            y2 = mid_y - mins[x] * half_h
            painter.drawLine(int(rect.left() + x), int(y1), int(rect.left() + x), int(y2))

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Масштабирование (Zoom In / Zoom Out) вокруг позиции курсора мыши."""
        total = max(1024, self._audio.shape[0])
        factor = 0.8 if event.angleDelta().y() > 0 else 1.25
        span = max(256, int((self.view_end_frame - self.view_start_frame) * factor))
        span = min(total, span)
        center_ratio = event.position().x() / max(1, self.width())
        anchor = self.view_start_frame + int((self.view_end_frame - self.view_start_frame) * center_ratio)
        self.view_start_frame = max(0, min(total - span, anchor - int(span * center_ratio)))
        self.view_end_frame = self.view_start_frame + span
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        frame = self._x_to_frame(event.position().x())
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_frame = frame
            self.playhead_frame = frame
            self.selection_range = None
            self.seekRequested.emit(frame)
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start_frame is not None:
            curr = self._x_to_frame(event.position().x())
            s, e = min(self._drag_start_frame, curr), max(self._drag_start_frame, curr)
            if e - s > 64:
                self.selection_range = (s, e)
                self.selectionChanged.emit(s, e)
                self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_start_frame = None

    def _x_to_frame(self, x: float) -> int:
        ratio = np.clip(x / max(1, self.width()), 0.0, 1.0)
        span = self.view_end_frame - self.view_start_frame
        return int(self.view_start_frame + ratio * span)


class LevelAndPhaseMeterWidget(QWidget):
    """Индикатор уровня (Peak + RMS в дБ) и стерео-фазовый коррелятор (-1 .. +1)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(180, 72)
        self.setAccessibleName("Измеритель уровня Peak/RMS и фазовый коррелятор")
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

        p.setFont(QFont("SansSerif", 8))
        p.setPen(QColor("#B0BEC5"))
        p.drawText(8, 14, f"Peak: {self.peak_db:5.1f} dB | RMS: {self.rms_db:5.1f} dB")

        p.fillRect(8, 20, w - 16, 14, QColor("#263238"))
        bar_color = QColor("#FF5252") if self.peak_db > -1.0 else QColor("#00E676")
        p.fillRect(8, 20, rms_w, 14, QColor("#00B0FF"))
        p.setPen(QPen(bar_color, 2.5))
        p.drawLine(8 + peak_w, 19, 8 + peak_w, 35)

        # Фазовый коррелятор [-1 .. +1]
        p.setPen(QColor("#B0BEC5"))
        p.drawText(8, 50, f"Phase Correlation: {self.phase:+.2f}")
        p.fillRect(8, 54, w - 16, 10, QColor("#263238"))
        center_x = w // 2
        phase_x = int(center_x + self.phase * ((w - 16) / 2.0))
        p.setPen(QPen(QColor("#FFC107"), 3.0))
        p.drawLine(center_x, 54, phase_x, 54)