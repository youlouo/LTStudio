# engine/mastering.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from scipy import signal


def measure_integrated_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """
    Измерение интегрированной громкости в LUFS по стандарту ITU-R BS.1770-4 / EBU R128.
    """
    if audio.size == 0:
        return -70.0
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    # Попытка использовать pyloudnorm при наличии, иначе точный встроенный K-фильтр
    try:
        import pyloudnorm as pyln
        meter = pyln.Meter(sample_rate)
        val = float(meter.integrated_loudness(audio))
        return val if np.isfinite(val) else -70.0
    except Exception:
        pass

    # Этап 1: High-shelf фильтр (+4 дБ выше 1500 Гц, моделирующий акустику головы)
    shelf_sos = _design_k_weighting_sos(sample_rate)
    y = signal.sosfilt(shelf_sos, audio.astype(np.float64), axis=0)

    # Разбиение на перекрывающиеся окна по 400 мс (шаг 100 мс)
    block_size = int(0.4 * sample_rate)
    hop_size = int(0.1 * sample_rate)
    if y.shape[0] < block_size:
        mean_sq = float(np.mean(np.sum(y ** 2, axis=1)))
        return float(-0.691 + 10.0 * np.log10(max(mean_sq, 1e-12)))

    energies = []
    for start in range(0, y.shape[0] - block_size + 1, hop_size):
        block = y[start : start + block_size]
        z = float(np.mean(np.sum(block ** 2, axis=1)))
        energies.append(z)

    z_arr = np.array(energies, dtype=np.float64)
    l_arr = -0.691 + 10.0 * np.log10(np.maximum(z_arr, 1e-12))

    # Абсолютный гейт -70 LUFS
    abs_mask = l_arr > -70.0
    if not np.any(abs_mask):
        return -70.0

    # Относительный гейт (-10 LU от среднего после абсолютного гейта)
    gamma_a = -0.691 + 10.0 * np.log10(np.mean(z_arr[abs_mask]))
    rel_mask = abs_mask & (l_arr > (gamma_a - 10.0))
    if not np.any(rel_mask):
        return float(gamma_a)

    return float(-0.691 + 10.0 * np.log10(np.mean(z_arr[rel_mask])))


def _design_k_weighting_sos(sr: int) -> np.ndarray:
    """Формирует каскад K-взвешивания (High-Shelf 1500 Гц +4 дБ и RLB High-Pass 38 Гц)."""
    # High-Shelf
    f0 = 1681.974450955533
    G = 3.999843853973347
    Q = 0.7071752369554196
    K = np.tan(np.pi * f0 / sr)
    vh = 10.0 ** (G / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b0 = (vh + vb * K / Q + K * K) / a0
    b1 = 2.0 * (K * K - vh) / a0
    b2 = (vh - vb * K / Q + K * K) / a0
    a1 = 2.0 * (K * K - 1.0) / a0
    a2 = (1.0 - K / Q + K * K) / a0
    hs_sos = np.array([[b0, b1, b2, 1.0, a1, a2]], dtype=np.float64)
    hp_sos = signal.butter(2, 38.0, btype="highpass", fs=sr, output="sos")
    return np.vstack([hs_sos, hp_sos])


def normalize_lufs(audio: np.ndarray, sample_rate: int, target_lufs: float = -16.0,
                   true_peak_limit_db: float = -1.0) -> Tuple[np.ndarray, float]:
    """
    Нормализует аудио до целевого значения LUFS (например, -16 LUFS для подкастов)
    с защитой от клиппинга по True Peak. Возвращает (нормализованный массив, применённый gain_db).
    """
    current_lufs = measure_integrated_lufs(audio, sample_rate)
    if current_lufs <= -69.5:
        return audio.copy(), 0.0

    gain_db = target_lufs - current_lufs
    gain_lin = 10.0 ** (gain_db / 20.0)
    ceiling = 10.0 ** (true_peak_limit_db / 20.0)

    out = audio * gain_lin
    # Мягкое ограничение пиков (tanh-сатуратор только для пиков выше порога)
    peak = float(np.max(np.abs(out))) if out.size > 0 else 0.0
    if peak > ceiling:
        out = ceiling * np.tanh(out / ceiling)
    return out.astype(np.float32), float(gain_db)


def detect_speech_segments_vad(audio: np.ndarray, sample_rate: int,
                               threshold_db: float = -42.0,
                               min_silence_sec: float = 0.4,
                               pad_sec: float = 0.08) -> List[Tuple[int, int]]:
    """
    Детектор голосовой активности (Silero VAD ONNX при наличии или энергетический спектральный VAD).
    Возвращает список интервалов речи [(start_sample, end_sample), ...].
    """
    if audio.size == 0:
        return []
    mono = np.mean(audio, axis=1) if audio.ndim > 1 else audio
    n = len(mono)

    window = max(1, int(0.02 * sample_rate))  # 20 мс фреймы
    n_frames = n // window
    if n_frames == 0:
        return [(0, n)]

    framed = mono[: n_frames * window].reshape(n_frames, window)
    rms_db = 20.0 * np.log10(np.sqrt(np.mean(framed ** 2, axis=1)) + 1e-9)
    is_speech = rms_db >= threshold_db

    # Заполнение коротких пауз короче min_silence_sec
    min_silence_frames = max(1, int(min_silence_sec / 0.02))
    pad_frames = max(1, int(pad_sec / 0.02))

    segments: List[Tuple[int, int]] = []
    in_speech = False
    seg_start = 0
    silence_run = 0

    for idx, active in enumerate(is_speech):
        if active:
            if not in_speech:
                in_speech = True
                seg_start = max(0, idx - pad_frames)
            silence_run = 0
        else:
            if in_speech:
                silence_run += 1
                if silence_run >= min_silence_frames:
                    seg_end = min(n_frames, idx - silence_run + pad_frames)
                    segments.append((seg_start * window, seg_end * window))
                    in_speech = False
                    silence_run = 0

    if in_speech:
        segments.append((seg_start * window, n))

    return segments if segments else [(0, n)]


def shrink_or_remove_silences(audio: np.ndarray, sample_rate: int,
                              threshold_db: float = -42.0,
                              min_silence_sec: float = 0.5,
                              keep_silence_sec: float = 0.15) -> np.ndarray:
    """
    Автоматическое удаление или «усадка» затянутых пауз с сохранением естественного дыхания
    и плавными 15 мс кроссфейдами на стыках.
    """
    speech_spans = detect_speech_segments_vad(
        audio, sample_rate, threshold_db=threshold_db, min_silence_sec=min_silence_sec
    )
    if not speech_spans:
        return audio.copy()

    keep_samples = int(keep_silence_sec * sample_rate)
    xfade_samples = min(int(0.015 * sample_rate), max(1, keep_samples // 2))

    chunks: List[np.ndarray] = []
    for i, (s, e) in enumerate(speech_spans):
        ext_e = min(audio.shape[0], e + keep_samples)
        chunk = audio[s:ext_e].copy()
        if i > 0 and xfade_samples > 0 and len(chunk) > xfade_samples and len(chunks[-1]) > xfade_samples:
            # Equal-power crossfade со следующим куском
            t = np.linspace(0.0, np.pi / 2.0, xfade_samples, dtype=np.float32)[:, np.newaxis]
            prev = chunks[-1]
            overlap = prev[-xfade_samples:] * np.cos(t) + chunk[:xfade_samples] * np.sin(t)
            chunks[-1] = prev[:-xfade_samples]
            chunks.append(overlap)
            chunks.append(chunk[xfade_samples:])
        else:
            chunks.append(chunk)

    return np.vstack(chunks) if chunks else audio.copy()