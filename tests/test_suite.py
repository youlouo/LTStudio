# tests/test_suite.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest

from engine.audio_engine import VoiceAudioEngine
from engine.crash_journal import CrashRecoveryJournal
from engine.dsp_chain import DSPEffectChain, ParametricEQ10BandPlugin
from engine.mastering import measure_integrated_lufs, normalize_lufs, shrink_or_remove_silences
from engine.transcription import TranscriptionEngine
from export.exporter import AudioExporter, EXPORT_PRESETS
from project.edl import NonDestructiveTimeline, TimelineRegion
from project.storage import ProjectRepository, ProjectMetadata


def _generate_synthetic_speech(sr: int = 48000, duration_sec: float = 2.0) -> np.ndarray:
    """Генерирует тестовый речеподобный сигнал (гармоники 180 Гц + сибилянты 6.8 кГц + пауза в середине)."""
    t = np.linspace(0.0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    carrier = 0.25 * np.sin(2.0 * np.pi * 180.0 * t) + 0.12 * np.sin(2.0 * np.pi * 360.0 * t)
    sibilance = 0.08 * np.sin(2.0 * np.pi * 6800.0 * t)
    sig = carrier + sibilance
    # Искусственная пауза с 0.8 по 1.4 секунду
    sig[int(0.8 * sr) : int(1.4 * sr)] = 0.0
    return sig[:, np.newaxis]


def test_non_destructive_edl_and_branching_history():
    sr = 48000
    tl = NonDestructiveTimeline(sample_rate=sr, channels=1)
    source_audio = _generate_synthetic_speech(sr, duration_sec=2.0)
    original_copy = source_audio.copy()

    tl.register_source("mic_take_1", source_audio, append_to_timeline=True)
    assert tl.total_frames == 2 * sr

    # Создаем снапшот исходного состояния
    tl.create_snapshot("До монтажа")

    # Удаляем 0.5 секунды из середины по словам транскрипта с авто-кроссфейдом
    tl.delete_transcript_words([(int(0.5 * sr), int(1.0 * sr))], crossfade_ms=15.0)
    after_cut_frames = tl.total_frames
    assert after_cut_frames < int(1.5 * sr)

    # Проверяем, что исходный буфер в памяти остался абсолютно неизменным (Non-Destructive)
    np.testing.assert_array_equal(tl._sources["mic_take_1"], original_copy)

    # Проверяем Undo / Redo и восстановление снапшота
    assert tl.undo() is True
    assert tl.total_frames == 2 * sr
    assert tl.redo() is True
    assert tl.total_frames == after_cut_frames
    assert tl.restore_snapshot("До монтажа") is True
    assert tl.total_frames == 2 * sr


def test_dsp_chain_continuity_and_ab_switching():
    sr = 48000
    chain = DSPEffectChain()
    eq: ParametricEQ10BandPlugin = chain.slot_a[3]  # 10-полосный EQ
    eq.set_band(5, freq=1000.0, gain_db=6.0, q=1.4)

    chain.reset_all(sr, 1)
    signal_in = _generate_synthetic_speech(sr, duration_sec=1.0)

    # Потоковая обработка блоками по 1024 сэмпла
    out_blocks = []
    for start in range(0, len(signal_in), 1024):
        out_blocks.append(chain.process(signal_in[start : start + 1024], sr))
    streamed_out = np.vstack(out_blocks)

    assert streamed_out.shape == signal_in.shape
    assert np.all(np.isfinite(streamed_out))

    # Проверка BYPASS режима
    chain.active_slot = "BYPASS"
    bypassed = chain.process(signal_in[:1024], sr)
    np.testing.assert_allclose(bypassed, signal_in[:1024])


def test_lufs_normalization_and_vad_silence_shrink():
    sr = 48000
    speech = _generate_synthetic_speech(sr, duration_sec=3.0) * 0.1
    norm_audio, gain_db = normalize_lufs(speech, sr, target_lufs=-16.0, true_peak_limit_db=-1.0)
    measured = measure_integrated_lufs(norm_audio, sr)

    assert gain_db > 0.0
    assert abs(measured - (-16.0)) < 1.2
    assert float(np.max(np.abs(norm_audio))) <= 10.0 ** (-1.0 / 20.0) + 1e-3

    # Проверка усадки паузы
    compacted = shrink_or_remove_silences(speech, sr, min_silence_sec=0.3, keep_silence_sec=0.1)
    assert compacted.shape[0] < speech.shape[0]


def test_full_integration_record_crash_recover_and_export(tmp_path: Path):
    # 1. Запись в 96 кГц / 24 бит без дропов
    engine = VoiceAudioEngine(tmp_path)
    engine.configure_audio(sample_rate=96000, channels=1, bit_depth="PCM_24")
    engine.start_recording()

    test_block = _generate_synthetic_speech(sr=96000, duration_sec=0.5)
    engine.push_input_block(test_block)
    marker = engine.add_recording_marker("Важная цитата")
    assert marker["label"] == "Важная цитата"

    saved_wav = engine.stop_recording(tmp_path / "interview_96k.wav")
    assert saved_wav.exists()
    assert engine.timeline.total_frames == test_block.shape[0]

    # 2. Эмуляция внезапного сбоя питания во время второй сессии записи
    crash_journal = CrashRecoveryJournal(tmp_path / "journals")
    crash_journal.start_session(48000, 1)
    crash_journal.append_block(_generate_synthetic_speech(48000, 0.4))
    crash_journal._write_atomic_metadata(status="RECORDING")
    # Не вызываем finalize_to_audio_file, имитируя SIGKILL
    recovered_list = CrashRecoveryJournal.recover_incomplete_sessions(tmp_path / "journals")
    assert len(recovered_list) == 1
    assert recovered_list[0].exists()

    # 3. Транскрипция, экспорт SRT и пакетный экспорт регионов с пресетом Подкаст (-16 LUFS)
    transcriber = TranscriptionEngine()
    words = transcriber.transcribe_audio(engine.timeline.render_timeline(), engine.sample_rate)
    srt_text = transcriber.export_subtitles(words, fmt="SRT")
    assert "-->" in srt_text

    engine.timeline.regions.append(
        TimelineRegion(region_id="r1", name="Интро подкаста", start_frame=0, end_frame=24000)
    )
    exported_regions = AudioExporter.export_regions_batch(
        engine.timeline, engine.dsp_chain, tmp_path / "exports", fmt="WAV", preset=EXPORT_PRESETS["podcast_mono"]
    )
    assert len(exported_regions) == 1 and exported_regions[0].exists()

    # 4. Сохранение в SQLite и упаковка в единый архив .voxproj
    repo = ProjectRepository(tmp_path / "db.sqlite")
    meta = ProjectMetadata(project_id="p_01", title="Выпуск 1", speaker="Алексей", tags=["интервью", "96k"])
    repo.save_project(meta, engine.timeline.to_dict(), engine.dsp_chain.to_dict(), [])
    bundle = repo.export_project_bundle("p_01", [saved_wav], tmp_path / "backup.ltproj")
    assert bundle.exists() and bundle.suffix == ".ltproj"