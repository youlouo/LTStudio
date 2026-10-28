# engine/dsp_chain.py
from __future__ import annotations
import abc
import copy
from typing import Dict, List, Any, Optional
import numpy as np
from scipy import signal


class AudioEffectPlugin(abc.ABC):
    """
    Базовый интерфейс плагина цифровой обработки сигнала (DSP).
    Поддерживает потоковую обработку без разрывов фазы на границах блоков.
    """
    plugin_id: str = "base"
    display_name: str = "Base Plugin"
    description: str = "Базовый модуль обработки аудиосигнала."
    primary_param: str = "intensity"
    primary_label: str = "Интенсивность (%)"
    primary_range: tuple[float, float] = (0.0, 100.0)
    is_vocal_only: bool = False

    def __init__(self, enabled: bool = True) -> None:
        self.enabled: bool = enabled
        self.params: Dict[str, Any] = {"intensity": 50.0}

    @abc.abstractmethod
    def reset_state(self, sample_rate: int, channels: int) -> None:
        pass

    @abc.abstractmethod
    def process_block(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        pass

    def get_primary_value(self) -> float:
        return float(self.params.get(self.primary_param, 50.0))

    def set_primary_value(self, value: float) -> None:
        lo, hi = self.primary_range
        self.params[self.primary_param] = float(np.clip(value, lo, hi))
        self.on_param_changed()

    def on_param_changed(self) -> None:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "enabled": self.enabled,
            "params": copy.deepcopy(self.params),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], registry: Dict[str, type]) -> Optional[AudioEffectPlugin]:
        pid = data.get("plugin_id")
        if pid not in registry:
            return None
        plugin_cls = registry[pid]
        instance = plugin_cls(enabled=data.get("enabled", True))
        for k, v in data.get("params", {}).items():
            instance.set_param(k, v)
        instance.on_param_changed()
        return instance

    def set_param(self, key: str, value: Any) -> None:
        self.params[key] = value


def _design_peaking_biquad(freq: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    freq = np.clip(freq, 20.0, sr * 0.48)
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sr
    alpha = np.sin(w0) / (2.0 * max(q, 0.1))
    cos_w0 = np.cos(w0)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


class DeClickDePopPlugin(AudioEffectPlugin):
    plugin_id = "de_click_pop"
    display_name = "Подавление взрывных согласных и щелчков (De-Pop / De-Click)"
    description = (
        "Фильтр верхних частот (85 Гц) устраняет низкочастотные удары воздуха от звуков Б/П,\n"
        "а медианный детектор сглаживает импульсные щелчки слюны и губ."
    )
    primary_param = "plosive_cut_hz"
    primary_label = "Частота среза взрывных (Гц)"
    primary_range = (40.0, 150.0)

    def __init__(self, enabled: bool = True) -> None:
        super().__init__(enabled)
        self.params = {"plosive_cut_hz": 85.0, "click_sensitivity": 4.5}
        self._hp_sos: Optional[np.ndarray] = None
        self._hp_zi: Optional[np.ndarray] = None
        self._sr: int = 0
        self._ch: int = 0

    def on_param_changed(self) -> None:
        if self._sr > 0 and self._ch > 0:
            self.reset_state(self._sr, self._ch)

    def reset_state(self, sample_rate: int, channels: int) -> None:
        self._sr = sample_rate
        self._ch = channels
        cutoff = np.clip(self.params["plosive_cut_hz"], 20.0, 200.0)
        self._hp_sos = signal.butter(4, cutoff, btype="highpass", fs=sample_rate, output="sos")
        self._hp_zi = np.zeros((self._hp_sos.shape[0], 2, channels), dtype=np.float64)

    def process_block(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or audio.size == 0:
            return audio
        if sample_rate != self._sr or audio.shape[1] != self._ch or self._hp_sos is None:
            self.reset_state(sample_rate, audio.shape[1])

        filtered, self._hp_zi = signal.sosfilt(
            self._hp_sos, audio.astype(np.float64), axis=0, zi=self._hp_zi
        )
        out = filtered.astype(np.float32)

        if out.shape[0] > 8:
            diff = np.abs(np.diff(out, n=2, axis=0))
            mad = np.median(diff, axis=0) + 1e-6
            spikes = diff > (self.params["click_sensitivity"] * 4.0 * mad)
            spike_indices = np.where(np.any(spikes, axis=1))[0] + 1
            for idx in spike_indices:
                if 2 <= idx < out.shape[0] - 2:
                    out[idx] = 0.5 * (out[idx - 2] + out[idx + 2])
        return out


class SpectralDenoisePlugin(AudioEffectPlugin):
    plugin_id = "spectral_denoise"
    display_name = "Спектральное шумоподавление (Spectral Denoise)"
    description = (
        "Анализирует частотный спектр сигнала и подавляет постоянный фоновый шум комнаты,\n"
        "гул вентиляторов ПК и шипение микрофонного предусилителя."
    )
    primary_param = "reduction_db"
    primary_label = "Подавление шума (дБ)"
    primary_range = (3.0, 30.0)

    def __init__(self, enabled: bool = False) -> None:
        super().__init__(enabled)
        self.params = {"reduction_db": 14.0, "noise_floor_db": -52.0}
        self._noise_profile: Optional[np.ndarray] = None

    def reset_state(self, sample_rate: int, channels: int) -> None:
        pass

    def process_block(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or audio.shape[0] < 512:
            return audio
        atten = 10.0 ** (-float(self.params["reduction_db"]) / 20.0)
        floor_lin = 10.0 ** (float(self.params["noise_floor_db"]) / 20.0)

        out = np.empty_like(audio)
        for ch in range(audio.shape[1]):
            _, _, zxx = signal.stft(audio[:, ch], fs=sample_rate, nperseg=512)
            mag = np.abs(zxx)
            thresh = np.percentile(mag, 25, axis=1, keepdims=True) + floor_lin
            mask = np.where(mag > thresh, 1.0, atten)
            mask = signal.convolve2d(mask, np.ones((3, 3)) / 9.0, mode="same")
            _, rec = signal.istft(zxx * mask, fs=sample_rate, nperseg=512)
            out[:, ch] = rec[: audio.shape[0]].astype(np.float32)
        return out


class NoiseGatePlugin(AudioEffectPlugin):
    plugin_id = "noise_gate"
    display_name = "Шумовой гейт (Noise Gate)"
    description = (
        "Автоматически приглушает звук в паузах между словами, когда уровень сигнала\n"
        "падает ниже установленного порога."
    )
    primary_param = "threshold_db"
    primary_label = "Порог срабатывания (дБ)"
    primary_range = (-70.0, -20.0)

    def __init__(self, enabled: bool = True) -> None:
        super().__init__(enabled)
        self.params = {
            "threshold_db": -45.0,
            "hysteresis_db": 4.0,
            "reduction_db": -60.0,
            "attack_ms": 5.0,
            "release_ms": 80.0,
        }
        self._env: float = 0.0
        self._gate_open: bool = False

    def reset_state(self, sample_rate: int, channels: int) -> None:
        self._env = 0.0
        self._gate_open = False

    def process_block(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or audio.size == 0:
            return audio
        open_lin = 10.0 ** (self.params["threshold_db"] / 20.0)
        close_lin = 10.0 ** ((self.params["threshold_db"] - self.params["hysteresis_db"]) / 20.0)
        floor_gain = 10.0 ** (self.params["reduction_db"] / 20.0)

        peak_env = np.max(np.abs(audio), axis=1)
        block_rms = float(np.sqrt(np.mean(peak_env ** 2) + 1e-12))

        if self._gate_open and block_rms < close_lin:
            self._gate_open = False
        elif not self._gate_open and block_rms >= open_lin:
            self._gate_open = True

        target_gain = 1.0 if self._gate_open else floor_gain
        tau_ms = self.params["attack_ms"] if target_gain > self._env else self.params["release_ms"]
        alpha = float(np.exp(-1.0 / max(1.0, (tau_ms * 0.001 * sample_rate))))

        n = audio.shape[0]
        gains = np.empty(n, dtype=np.float32)
        g = self._env
        for i in range(n):
            g = alpha * g + (1.0 - alpha) * target_gain
            gains[i] = g
        self._env = float(g)
        return audio * gains[:, np.newaxis]


class BreathControlPlugin(AudioEffectPlugin):
    plugin_id = "breath_control"
    display_name = "Смягчение вдохов (Breath Control)"
    description = (
        "Определяет среднечастотные шумовые всплески между фразами (вдохи и выдохи)\n"
        "и мягко снижает их громкость, сохраняя естественность речи."
    )
    primary_param = "sensitivity"
    primary_label = "Степень ослабления вдохов (%)"
    primary_range = (0.0, 100.0)

    def __init__(self, enabled: bool = False) -> None:
        super().__init__(enabled)
        self.params = {"sensitivity": 60.0}

    def reset_state(self, sample_rate: int, channels: int) -> None:
        pass

    def process_block(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or audio.size == 0:
            return audio
        sens = float(self.params.get("sensitivity", 60.0)) / 100.0
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        rms_db = 20.0 * np.log10(rms)
        if -42.0 <= rms_db <= -24.0:
            atten = 1.0 - 0.65 * sens
            return (audio * atten).astype(np.float32)
        return audio


class BassBoostPlugin(AudioEffectPlugin):
    plugin_id = "bass_boost"
    display_name = "Наложение баса (Sub-Bass & Low Boost)"
    description = (
        "Насыщает голос глубокими низкими частотами (95–140 Гц) и легкими субгармониками.\n"
        "Придает мужскому и женскому голосу плотный «дикторский» бархатный тембр."
    )
    primary_param = "bass_db"
    primary_label = "Усиление баса (дБ)"
    primary_range = (0.0, 15.0)

    def __init__(self, enabled: bool = False) -> None:
        super().__init__(enabled)
        self.params = {"bass_db": 6.0, "freq_hz": 115.0}
        self._sos: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None
        self._sr: int = 0
        self._ch: int = 0

    def on_param_changed(self) -> None:
        if self._sr > 0 and self._ch > 0:
            self.reset_state(self._sr, self._ch)

    def reset_state(self, sample_rate: int, channels: int) -> None:
        self._sr = sample_rate
        self._ch = channels
        gain = float(self.params.get("bass_db", 6.0))
        freq = float(self.params.get("freq_hz", 115.0))
        self._sos = _design_peaking_biquad(freq, gain, q=0.85, sr=sample_rate)
        self._zi = np.zeros((self._sos.shape[0], 2, channels), dtype=np.float64)

    def process_block(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or audio.size == 0:
            return audio
        if sample_rate != self._sr or audio.shape[1] != self._ch or self._sos is None:
            self.reset_state(sample_rate, audio.shape[1])
        boosted, self._zi = signal.sosfilt(self._sos, audio.astype(np.float64), axis=0, zi=self._zi)
        return np.tanh(boosted).astype(np.float32)


class VoiceEnhancerPlugin(AudioEffectPlugin):
    plugin_id = "voice_enhancer"
    display_name = "Улучшение голоса (Studio Voice Enhancer)"
    description = (
        "Комплексный эксайтер речи: вырезает коробочный призвук комнаты (320 Гц),\n"
        "поднимает область артикуляции (3.2 кГц) и добавляет студийный «воздух» (10 кГц)."
    )
    primary_param = "clarity"
    primary_label = "Ясность и присутствие (%)"
    primary_range = (0.0, 100.0)

    def __init__(self, enabled: bool = True) -> None:
        super().__init__(enabled)
        self.params = {"clarity": 55.0}
        self._sos: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None
        self._sr: int = 0
        self._ch: int = 0

    def on_param_changed(self) -> None:
        if self._sr > 0 and self._ch > 0:
            self.reset_state(self._sr, self._ch)

    def reset_state(self, sample_rate: int, channels: int) -> None:
        self._sr = sample_rate
        self._ch = channels
        amount = float(self.params.get("clarity", 55.0)) / 100.0
        s1 = _design_peaking_biquad(320.0, -3.5 * amount, 1.4, sample_rate)
        s2 = _design_peaking_biquad(3200.0, 4.5 * amount, 1.1, sample_rate)
        s3 = _design_peaking_biquad(9500.0, 5.0 * amount, 0.9, sample_rate)
        self._sos = np.vstack([s1, s2, s3])
        self._zi = np.zeros((self._sos.shape[0], 2, channels), dtype=np.float64)

    def process_block(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or audio.size == 0:
            return audio
        if sample_rate != self._sr or audio.shape[1] != self._ch or self._sos is None:
            self.reset_state(sample_rate, audio.shape[1])
        out, self._zi = signal.sosfilt(self._sos, audio.astype(np.float64), axis=0, zi=self._zi)
        return out.astype(np.float32)


class LiveSoundPlugin(AudioEffectPlugin):
    plugin_id = "live_sound"
    display_name = "Живой звук (Warmth & Live Saturation)"
    description = (
        "Мягкое аналоговое насыщение четными гармониками в сочетании с микро-отражениями.\n"
        "Убирает цифровую «сухость» микрофона и делает голос живым и теплым."
    )
    primary_param = "drive"
    primary_label = "Теплота и насыщение (%)"
    primary_range = (0.0, 100.0)

    def __init__(self, enabled: bool = False) -> None:
        super().__init__(enabled)
        self.params = {"drive": 45.0}

    def reset_state(self, sample_rate: int, channels: int) -> None:
        pass

    def process_block(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or audio.size == 0:
            return audio
        drv = 1.0 + (float(self.params.get("drive", 45.0)) / 100.0) * 2.2
        # Мягкий асимметричный ламповый сатуратор
        saturated = np.tanh(audio * drv) / np.tanh(drv)
        even_harm = 0.04 * (audio ** 2) * np.sign(audio)
        return np.clip(saturated + even_harm, -0.99, 0.99).astype(np.float32)


class ParametricEQ10BandPlugin(AudioEffectPlugin):
    plugin_id = "parametric_eq_10"
    display_name = "Параметрический эквалайзер (10 полос)"
    description = "10-полосный точный эквалайзер для частотной коррекции тембра."
    primary_param = "presence_gain"
    primary_label = "Усиление средних частот (дБ)"
    primary_range = (-12.0, 12.0)

    DEFAULT_FREQS = [31.5, 63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]

    def __init__(self, enabled: bool = True) -> None:
        super().__init__(enabled)
        self.params = {
            "presence_gain": 0.0,
            "bands": [{"freq": f, "gain_db": 0.0, "q": 1.414} for f in self.DEFAULT_FREQS],
        }
        self._sos: Optional[np.ndarray] = None
        self._zi: Optional[np.ndarray] = None
        self._last_sr: int = 0
        self._last_channels: int = 0

    def on_param_changed(self) -> None:
        val = float(self.params.get("presence_gain", 0.0))
        self.params["bands"][6]["gain_db"] = val
        self._rebuild_filters()

    def set_band(self, index: int, freq: Optional[float] = None,
                 gain_db: Optional[float] = None, q: Optional[float] = None) -> None:
        band = self.params["bands"][index]
        if freq is not None:
            band["freq"] = float(freq)
        if gain_db is not None:
            band["gain_db"] = float(np.clip(gain_db, -24.0, 24.0))
        if q is not None:
            band["q"] = float(np.clip(q, 0.1, 18.0))
        self._rebuild_filters()

    def _rebuild_filters(self) -> None:
        if self._last_sr <= 0 or self._last_channels <= 0:
            return
        sos_list = []
        for band in self.params["bands"]:
            if abs(band["gain_db"]) > 1e-3:
                sos_list.append(_design_peaking_biquad(band["freq"], band["gain_db"], band["q"], self._last_sr))
        if sos_list:
            self._sos = np.vstack(sos_list)
            self._zi = np.zeros((self._sos.shape[0], 2, self._last_channels), dtype=np.float64)
        else:
            self._sos = None
            self._zi = None

    def reset_state(self, sample_rate: int, channels: int) -> None:
        self._last_sr = sample_rate
        self._last_channels = channels
        self._rebuild_filters()

    def process_block(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or audio.size == 0:
            return audio
        if sample_rate != self._last_sr or audio.shape[1] != self._last_channels:
            self.reset_state(sample_rate, audio.shape[1])
        if self._sos is None or self._zi is None:
            return audio
        out, self._zi = signal.sosfilt(self._sos, audio.astype(np.float64), axis=0, zi=self._zi)
        return out.astype(np.float32)


class DeEsserPlugin(AudioEffectPlugin):
    plugin_id = "de_esser"
    display_name = "Подавление сибилянтов (De-Esser)"
    description = "Динамически смягчает резкие свистящие и шипящие звуки (С, З, Ц, Щ) в диапазоне 5–9 кГц."
    primary_param = "max_reduction_db"
    primary_label = "Степень подавления «С/Щ» (дБ)"
    primary_range = (0.0, 18.0)

    def __init__(self, enabled: bool = True) -> None:
        super().__init__(enabled)
        self.params = {
            "center_freq": 6800.0,
            "bandwidth_q": 1.6,
            "threshold_db": -24.0,
            "max_reduction_db": 10.0,
        }
        self._bp_sos: Optional[np.ndarray] = None
        self._bp_zi: Optional[np.ndarray] = None
        self._sr: int = 0
        self._ch: int = 0

    def reset_state(self, sample_rate: int, channels: int) -> None:
        self._sr = sample_rate
        self._ch = channels
        low = max(1000.0, self.params["center_freq"] * 0.7)
        high = min(sample_rate * 0.47, self.params["center_freq"] * 1.35)
        self._bp_sos = signal.butter(2, [low, high], btype="bandpass", fs=sample_rate, output="sos")
        self._bp_zi = np.zeros((self._bp_sos.shape[0], 2, channels), dtype=np.float64)

    def process_block(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or audio.size == 0:
            return audio
        if sample_rate != self._sr or audio.shape[1] != self._ch or self._bp_sos is None:
            self.reset_state(sample_rate, audio.shape[1])

        sibilance, self._bp_zi = signal.sosfilt(
            self._bp_sos, audio.astype(np.float64), axis=0, zi=self._bp_zi
        )
        sib_env = np.max(np.abs(sibilance), axis=1) + 1e-9
        sib_db = 20.0 * np.log10(sib_env)
        over_db = np.maximum(0.0, sib_db - self.params["threshold_db"])
        reduction_db = np.minimum(self.params["max_reduction_db"], over_db * 0.75)
        atten = 1.0 - (10.0 ** (-reduction_db / 20.0))
        return audio - (sibilance.astype(np.float32) * atten[:, np.newaxis])


class CompressorLimiterPlugin(AudioEffectPlugin):
    plugin_id = "compressor_limiter"
    display_name = "Компрессор и Лимитер (Broadcast Dynamics)"
    description = (
        "Выравнивает динамический диапазон речи: делает тихие слова громче,\n"
        "сдерживает громкие пики и защищает мастер-канал от перегрузок."
    )
    primary_param = "makeup_db"
    primary_label = "Компенсация громкости (Make-Up дБ)"
    primary_range = (0.0, 15.0)

    def __init__(self, enabled: bool = True) -> None:
        super().__init__(enabled)
        self.params = {
            "threshold_db": -18.0,
            "ratio": 3.5,
            "attack_ms": 10.0,
            "release_ms": 120.0,
            "makeup_db": 3.0,
            "ceiling_db": -1.0,
        }
        self._env_db: float = -96.0

    def reset_state(self, sample_rate: int, channels: int) -> None:
        self._env_db = -96.0

    def process_block(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or audio.size == 0:
            return audio
        thr = float(self.params["threshold_db"])
        ratio = max(1.0, float(self.params["ratio"]))
        makeup_lin = 10.0 ** (float(self.params["makeup_db"]) / 20.0)
        ceiling_lin = 10.0 ** (float(self.params["ceiling_db"]) / 20.0)

        step = 64
        n = audio.shape[0]
        gain_curve = np.ones(n, dtype=np.float32)

        att_coeff = float(np.exp(-step / max(1.0, self.params["attack_ms"] * 0.001 * sample_rate)))
        rel_coeff = float(np.exp(-step / max(1.0, self.params["release_ms"] * 0.001 * sample_rate)))

        env = self._env_db
        for start in range(0, n, step):
            end = min(start + step, n)
            chunk = audio[start:end]
            peak = float(np.max(np.abs(chunk)) + 1e-9)
            lvl_db = 20.0 * np.log10(peak)
            if lvl_db > env:
                env = att_coeff * env + (1.0 - att_coeff) * lvl_db
            else:
                env = rel_coeff * env + (1.0 - rel_coeff) * lvl_db

            over_db = env - thr
            if over_db > 0.0:
                gr_db = -over_db * (1.0 - 1.0 / ratio)
                gain_curve[start:end] = 10.0 ** (gr_db / 20.0)

        self._env_db = env
        out = audio * gain_curve[:, np.newaxis] * makeup_lin
        return np.clip(out, -ceiling_lin, ceiling_lin).astype(np.float32)


class EchoDelayPlugin(AudioEffectPlugin):
    plugin_id = "echo_delay"
    display_name = "Эхо и задержка (Echo & Delay)"
    description = (
        "Создает отчетливые затухающие повторы голоса (эффект просторного зала или художественного эха).\n"
        "Регулятор задает громкость эхо-повторов."
    )
    primary_param = "wet_mix"
    primary_label = "Уровень эха (%)"
    primary_range = (5.0, 75.0)

    def __init__(self, enabled: bool = False) -> None:
        super().__init__(enabled)
        self.params = {"delay_ms": 260.0, "feedback": 0.35, "wet_mix": 25.0}

    def reset_state(self, sample_rate: int, channels: int) -> None:
        pass

    def process_block(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or audio.size == 0:
            return audio
        delay_samples = max(1, int(float(self.params.get("delay_ms", 260.0)) * 0.001 * sample_rate))
        fb = float(np.clip(self.params.get("feedback", 0.35), 0.0, 0.85))
        wet = float(np.clip(self.params.get("wet_mix", 25.0) / 100.0, 0.0, 1.0))

        out = audio.astype(np.float32).copy()
        n = out.shape[0]
        for tap in range(1, 5):
            offset = delay_samples * tap
            gain = wet * (fb ** (tap - 1))
            if offset < n:
                out[offset:] += audio[: n - offset] * gain
        return np.clip(out, -0.99, 0.99)


class StereoStagePlugin(AudioEffectPlugin):
    plugin_id = "stereo_stage"
    display_name = "Сцена и объем (Soundstage & Ambience)"
    description = (
        "Формирует широкую акустическую сцену вокруг слушателя с помощью микро-задержек Хааса\n"
        "и ранних отражений студийного помещения."
    )
    primary_param = "width_pct"
    primary_label = "Ширина сцены (%)"
    primary_range = (0.0, 100.0)

    def __init__(self, enabled: bool = False) -> None:
        super().__init__(enabled)
        self.params = {"width_pct": 50.0}

    def reset_state(self, sample_rate: int, channels: int) -> None:
        pass

    def process_block(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or audio.size == 0:
            return audio
        width = float(self.params.get("width_pct", 50.0)) / 100.0
        haas_samples = max(1, int(0.011 * sample_rate))  # 11 мс эффект Хааса
        out = audio.copy()
        if out.shape[0] > haas_samples:
            side = np.zeros_like(out)
            side[haas_samples:] = out[:-haas_samples] * (0.28 * width)
            side[:-haas_samples] -= out[haas_samples:] * (0.18 * width)
            out = np.clip(out + side, -0.99, 0.99)
        return out.astype(np.float32)


class VoiceReverbDelayPlugin(AudioEffectPlugin):
    plugin_id = "voice_reverb"
    display_name = "Реверберация комнаты (Studio Room Reverb)"
    description = "Имитирует акустику студийного тон-зала для дикторской озвучки и вокала."
    primary_param = "wet_pct"
    primary_label = "Глубина реверберации (%)"
    primary_range = (0.0, 60.0)

    def __init__(self, enabled: bool = False) -> None:
        super().__init__(enabled)
        self.params = {"room_size": 0.35, "wet_mix": 0.14, "wet_pct": 14.0, "predelay_ms": 18.0}

    def on_param_changed(self) -> None:
        self.params["wet_mix"] = float(self.params.get("wet_pct", 14.0)) / 100.0

    def reset_state(self, sample_rate: int, channels: int) -> None:
        pass

    def process_block(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        wet = float(np.clip(self.params.get("wet_mix", 0.14), 0.0, 1.0))
        if not self.enabled or audio.size == 0 or wet <= 1e-4:
            return audio
        predelay_samples = int(self.params["predelay_ms"] * 0.001 * sample_rate)

        ir_len = max(64, int(sample_rate * max(0.05, self.params["room_size"] * 0.4)))
        t = np.linspace(0.0, 1.0, ir_len, dtype=np.float32)
        rng = np.random.default_rng(seed=42)
        ir = (rng.standard_normal(ir_len).astype(np.float32)) * np.exp(-6.0 * t)
        ir /= (np.linalg.norm(ir) + 1e-9)

        out = np.empty_like(audio)
        for ch in range(audio.shape[1]):
            conv = signal.fftconvolve(audio[:, ch], ir, mode="full")[: audio.shape[0]]
            if 0 < predelay_samples < conv.shape[0]:
                conv = np.concatenate([np.zeros(predelay_samples, dtype=np.float32), conv[:-predelay_samples]])
            out[:, ch] = (1.0 - wet) * audio[:, ch] + wet * conv
        return out


class RadioFilterPlugin(AudioEffectPlugin):
    plugin_id = "radio_filter"
    display_name = "Эффект рации / телефона (Bandpass Lo-Fi)"
    description = (
        "Сужает частотный диапазон до 400–3400 Гц с легким перегрузом,\n"
        "имитируя звучание телефонной линии, рации или ретро-радиоприемника."
    )
    primary_param = "mix_pct"
    primary_label = "Степень фильтрации (%)"
    primary_range = (0.0, 100.0)

    def __init__(self, enabled: bool = False) -> None:
        super().__init__(enabled)
        self.params = {"mix_pct": 100.0}

    def reset_state(self, sample_rate: int, channels: int) -> None:
        pass

    def process_block(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or audio.size == 0:
            return audio
        mix = float(self.params.get("mix_pct", 100.0)) / 100.0
        sos = signal.butter(4, [400.0, min(3400.0, sample_rate * 0.45)], btype="bandpass", fs=sample_rate, output="sos")
        filtered = signal.sosfilt(sos, audio.astype(np.float64), axis=0).astype(np.float32)
        crushed = np.tanh(filtered * 2.5)
        return (1.0 - mix) * audio + mix * crushed


class PitchTimeShiftPlugin(AudioEffectPlugin):
    plugin_id = "pitch_time_shift"
    display_name = "Сдвиг тона и темпа (Pitch & Time Shift)"
    description = "Изменяет высоту тона голоса в полутонах (от -12 до +12) с сохранением скорости."
    primary_param = "semitones"
    primary_label = "Сдвиг тона (полутона)"
    primary_range = (-12.0, 12.0)
    is_vocal_only = True

    def __init__(self, enabled: bool = False) -> None:
        super().__init__(enabled)
        self.params = {
            "semitones": 0.0,
            "time_rate": 1.0,
            "preserve_formants": True,
            "vocal_autotune_mode": False,
        }

    def reset_state(self, sample_rate: int, channels: int) -> None:
        pass

    def process_block(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.enabled or audio.size == 0:
            return audio
        semi = float(self.params["semitones"])
        rate = float(self.params["time_rate"])
        if abs(semi) < 1e-2 and abs(rate - 1.0) < 1e-2:
            return audio
        try:
            import pyrubberband as pyrb
            out = audio
            if abs(semi) >= 1e-2:
                out = pyrb.pitch_shift(out, sample_rate, semi)
            if abs(rate - 1.0) >= 1e-2:
                out = pyrb.time_stretch(out, sample_rate, rate)
            return out.astype(np.float32)
        except Exception:
            return self._phase_vocoder_process(audio, sample_rate, semi, rate)

    @staticmethod
    def _phase_vocoder_process(audio: np.ndarray, sr: int, semitones: float, rate: float) -> np.ndarray:
        pitch_factor = 2.0 ** (semitones / 12.0)
        effective_rate = rate / pitch_factor
        n_fft = 1024
        hop = n_fft // 4

        channels_out = []
        for ch in range(audio.shape[1]):
            x = audio[:, ch]
            if len(x) < n_fft:
                channels_out.append(x)
                continue
            _, _, stft_mat = signal.stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
            n_bins, n_frames = stft_mat.shape
            time_steps = np.arange(0, n_frames, effective_rate, dtype=np.float64)
            new_stft = np.zeros((n_bins, len(time_steps)), dtype=np.complex64)

            phase_acc = np.angle(stft_mat[:, 0])
            omega = 2.0 * np.pi * hop * np.arange(n_bins) / n_fft

            for i, step in enumerate(time_steps):
                idx = int(np.floor(step))
                frac = step - idx
                idx_next = min(idx + 1, n_frames - 1)
                mag = (1.0 - frac) * np.abs(stft_mat[:, idx]) + frac * np.abs(stft_mat[:, idx_next])
                dphase = np.angle(stft_mat[:, idx_next]) - np.angle(stft_mat[:, idx]) - omega
                dphase = (dphase + np.pi) % (2.0 * np.pi) - np.pi
                phase_acc += omega + dphase
                new_stft[:, i] = mag * np.exp(1j * phase_acc)

            _, stretched = signal.istft(new_stft, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
            if abs(pitch_factor - 1.0) > 1e-3:
                target_len = max(1, int(len(stretched) / pitch_factor))
                stretched = signal.resample(stretched, target_len)
            channels_out.append(stretched.astype(np.float32))

        min_len = min(len(c) for c in channels_out)
        return np.column_stack([c[:min_len] for c in channels_out])


PLUGIN_REGISTRY: Dict[str, type] = {
    DeClickDePopPlugin.plugin_id: DeClickDePopPlugin,
    SpectralDenoisePlugin.plugin_id: SpectralDenoisePlugin,
    NoiseGatePlugin.plugin_id: NoiseGatePlugin,
    BreathControlPlugin.plugin_id: BreathControlPlugin,
    BassBoostPlugin.plugin_id: BassBoostPlugin,
    VoiceEnhancerPlugin.plugin_id: VoiceEnhancerPlugin,
    LiveSoundPlugin.plugin_id: LiveSoundPlugin,
    ParametricEQ10BandPlugin.plugin_id: ParametricEQ10BandPlugin,
    DeEsserPlugin.plugin_id: DeEsserPlugin,
    CompressorLimiterPlugin.plugin_id: CompressorLimiterPlugin,
    EchoDelayPlugin.plugin_id: EchoDelayPlugin,
    StereoStagePlugin.plugin_id: StereoStagePlugin,
    VoiceReverbDelayPlugin.plugin_id: VoiceReverbDelayPlugin,
    RadioFilterPlugin.plugin_id: RadioFilterPlugin,
    PitchTimeShiftPlugin.plugin_id: PitchTimeShiftPlugin,
}


class DSPEffectChain:
    """Цепочка эффектов с поддержкой A/B сравнения и BYPASS."""

    def __init__(self) -> None:
        self.slot_a: List[AudioEffectPlugin] = self.create_default_voice_chain()
        self.slot_b: List[AudioEffectPlugin] = self.create_default_voice_chain()
        self.active_slot: str = "A"

    @staticmethod
    def create_default_voice_chain() -> List[AudioEffectPlugin]:
        return [
            DeClickDePopPlugin(enabled=True),
            SpectralDenoisePlugin(enabled=False),
            NoiseGatePlugin(enabled=True),
            BreathControlPlugin(enabled=False),
            BassBoostPlugin(enabled=False),
            VoiceEnhancerPlugin(enabled=True),
            LiveSoundPlugin(enabled=False),
            ParametricEQ10BandPlugin(enabled=True),
            DeEsserPlugin(enabled=True),
            CompressorLimiterPlugin(enabled=True),
            EchoDelayPlugin(enabled=False),
            StereoStagePlugin(enabled=False),
            VoiceReverbDelayPlugin(enabled=False),
            RadioFilterPlugin(enabled=False),
            PitchTimeShiftPlugin(enabled=False),
        ]

    @property
    def current_plugins(self) -> List[AudioEffectPlugin]:
        return self.slot_a if self.active_slot == "A" else self.slot_b

    def reset_all(self, sample_rate: int, channels: int) -> None:
        for p in self.slot_a + self.slot_b:
            p.reset_state(sample_rate, channels)

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.active_slot == "BYPASS" or audio.size == 0:
            return audio
        out = audio.copy()
        for plugin in self.current_plugins:
            if plugin.enabled:
                out = plugin.process_block(out, sample_rate)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_slot": self.active_slot,
            "slot_a": [p.to_dict() for p in self.slot_a],
            "slot_b": [p.to_dict() for p in self.slot_b],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DSPEffectChain:
        chain = cls()
        chain.active_slot = data.get("active_slot", "A")
        if "slot_a" in data:
            loaded_a = [AudioEffectPlugin.from_dict(d, PLUGIN_REGISTRY) for d in data["slot_a"]]
            chain.slot_a = [p for p in loaded_a if p is not None] or cls.create_default_voice_chain()
        if "slot_b" in data:
            loaded_b = [AudioEffectPlugin.from_dict(d, PLUGIN_REGISTRY) for d in data["slot_b"]]
            chain.slot_b = [p for p in loaded_b if p is not None] or cls.create_default_voice_chain()
        return chain