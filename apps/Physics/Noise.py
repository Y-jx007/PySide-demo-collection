# -*- coding: utf-8 -*-
"""Noise — real-time environmental noise statistics and long-term history.

The microphone path is intentionally lightweight: audio is captured in a
PortAudio callback into a lock-protected ring buffer.  The GUI only reads the
latest block at 5 Hz, computes A/Z levels and keeps one history sample per
second in SQLite.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
from matplotlib import rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from common.qt_base import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QTimer, QVBoxLayout, QWidget, Qt, sys,
)
from noise_history import NoiseHistory
from noise_dsp import (
    EPS, REFERENCE_SPL, a_weighting_db, analyse_power, fit_curve_db, leq,
    noise_colour, percentile_levels, reference_1f_db, spectral_slope,
    spectral_power, subtract_noise, third_octave_density,
    THIRD_OCTAVE_CENTERS,
)

rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "PingFang SC",
    "Noto Sans CJK SC", "WenQuanYi Micro Hei",
]
rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs"
DISPLAY_HZ = 10
DISPLAY_MS = 1000 // DISPLAY_HZ
DRAW_EVERY = 3
BLOCK_SIZE = 4096


def _spl_from_rms(rms, calibration_db=0.0):
    return REFERENCE_SPL + 20.0 * np.log10(max(float(rms), EPS)) + calibration_db


class AudioCapture:
    """PortAudio callback -> lock-protected ring buffer."""

    def __init__(self, fs, block_size, device=None):
        self.fs = int(fs)
        self.block_size = int(block_size)
        self.device = device
        # Keep every callback block until the GUI drains it.  Reading only
        # the newest ring block at 2 Hz used to discard ~80% of the audio and
        # miss short transients.
        self.pending = deque(maxlen=64)
        self.lock = threading.Lock()
        self.stream = None
        self.last_status = ""

    def _callback(self, indata, frames, _time_info, status):
        if status:
            self.last_status = str(status)
        mono = np.asarray(indata[:, 0] if indata.ndim > 1 else indata, dtype=np.float32)
        with self.lock:
            self.pending.append(mono.copy())

    def start(self):
        self.stream = sd.InputStream(
            device=self.device,
            samplerate=self.fs,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self.stream.start()

    def drain(self):
        with self.lock:
            if not self.pending:
                return None
            blocks = list(self.pending)
            self.pending.clear()
        return np.concatenate(blocks) if len(blocks) > 1 else blocks[0].copy()

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None


class DemoSource:
    """Synthetic source built from independently selectable components.

    1/f, 1/f^2, white noise and a transient are all optional.  The GUI defaults
    to 1/f only, but no component is structurally special.
    """

    def __init__(self, fs, include_half=False, include_pink=True,
                 include_one_half=False, include_brown=False,
                 include_white=False, include_transient=False):
        self.fs = int(fs)
        self.rng = np.random.default_rng(20261024)
        self.alphas = []
        if include_half:
            self.alphas.append(0.5)
        if include_pink:
            self.alphas.append(1.0)
        if include_one_half:
            self.alphas.append(1.5)
        if include_brown:
            self.alphas.append(2.0)
        self.include_white = bool(include_white)
        self.include_transient = bool(include_transient)
        self.enabled = bool(self.alphas or self.include_white or self.include_transient)
        self.t = 0.0

    def _colored(self, n, alpha):
        white = self.rng.normal(0.0, 1.0, n)
        spectrum = np.fft.rfft(white)
        bin_freq = np.fft.rfftfreq(n, 1.0 / self.fs)
        spectrum[1:] /= np.maximum(bin_freq[1:], 1.0) ** (alpha / 2.0)
        noise = np.fft.irfft(spectrum, n)
        noise *= 0.012 / max(float(np.std(noise)), 1e-9)
        return noise

    def block(self, n):
        t = self.t + np.arange(n, dtype=np.float64) / self.fs
        self.t += n / self.fs
        noise = np.zeros(n, dtype=np.float64)
        for alpha in self.alphas:
            noise += self._colored(n, alpha)
        if self.include_white:
            noise += 0.0035 * self.rng.normal(0.0, 1.0, n)
        if self.include_transient:
            phase = self.t % 9.0
            if phase < 0.8:
                noise += 0.025 * np.exp(-phase / 0.25) * np.sin(2 * np.pi * 180.0 * t)
        return noise


class NoiseWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Noise — 环境噪声实时统计")
        self.resize(1280, 840)
        self.setMinimumSize(980, 680)

        self.fs = 44100
        self.calibration = 0.0
        self.input_gain_db = 0.0
        self.silence_gate_db = -80.0
        self.denoise_strength = 0.8
        self.noise_psd = None
        self.running = True
        self.demo = False
        self.force_demo = os.environ.get("NOISE_DEMO") == "1"
        self.capture = None
        self.demo_source = DemoSource(self.fs)
        self.levels = deque(maxlen=DISPLAY_HZ * 60)
        self.band_centers = THIRD_OCTAVE_CENTERS.copy()
        self.avg_seconds = 30
        self.density_history = deque(maxlen=DISPLAY_HZ * 65)
        self.session_started = time.time()
        self.last_save = 0.0
        self.last_long_update = 0.0
        self.session_samples = 0
        self._draw_counter = 0

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.history = NoiseHistory(LOG_DIR / "noise_history.sqlite")

        self._build_ui()
        self._populate_devices()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.timer.start(DISPLAY_MS)
        QTimer.singleShot(150, self._start_monitor)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        left = QWidget()
        left.setFixedWidth(310)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        panel = QVBoxLayout(content)
        panel.setContentsMargins(0, 0, 0, 0)
        panel.setSpacing(8)
        panel.addWidget(self._build_input_group())
        panel.addWidget(self._build_calibration_group())
        panel.addWidget(self._build_stats_group())
        panel.addWidget(self._build_history_group())
        panel.addStretch(1)
        scroll.setWidget(content)
        left_layout.addWidget(scroll)
        root.addWidget(left, 0)

        self.figure = Figure(figsize=(9.2, 7.6), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.canvas, 1)

        self._build_axes()
        self.statusBar().showMessage("正在检测输入设备…")

    def _build_input_group(self):
        group = QGroupBox("输入")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        row = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setFixedWidth(54)
        row.addWidget(self.device_combo, 1)
        row.addWidget(self.refresh_btn)
        layout.addLayout(row)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("暂停")
        self.reset_btn = QPushButton("重置本次")
        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.reset_btn)
        layout.addLayout(buttons)

        self.demo_check = QCheckBox("演示模式（无麦克风）")
        layout.addWidget(self.demo_check)

        demo_opts = QHBoxLayout()
        self.half_check = QCheckBox("1/f^0.5")
        self.pink_check = QCheckBox("1/f")
        self.pink_check.setChecked(True)
        demo_opts.addWidget(self.half_check)
        demo_opts.addWidget(self.pink_check)
        layout.addLayout(demo_opts)

        demo_opts2 = QHBoxLayout()
        self.onehalf_check = QCheckBox("1/f^1.5")
        self.brown_check = QCheckBox("1/f^2")
        demo_opts2.addWidget(self.onehalf_check)
        demo_opts2.addWidget(self.brown_check)
        layout.addLayout(demo_opts2)

        demo_opts3 = QHBoxLayout()
        self.white_check = QCheckBox("白噪声")
        self.transient_check = QCheckBox("瞬态")
        demo_opts3.addWidget(self.white_check)
        demo_opts3.addWidget(self.transient_check)
        layout.addLayout(demo_opts3)

        avg_row = QHBoxLayout()
        avg_row.addWidget(QLabel("1/f 平均"))
        self.avg_combo = QComboBox()
        for label, seconds in (("5 s", 5), ("10 s", 10), ("30 s", 30), ("60 s", 60)):
            self.avg_combo.addItem(label, seconds)
        self.avg_combo.setCurrentIndex(2)
        avg_row.addWidget(self.avg_combo, 1)
        layout.addLayout(avg_row)

        self.mode_label = QLabel("未开始")
        self.mode_label.setStyleSheet("color:#777;")
        layout.addWidget(self.mode_label)

        self.refresh_btn.clicked.connect(self._refresh_devices)
        self.start_btn.clicked.connect(self._toggle_running)
        self.reset_btn.clicked.connect(self._reset_session)
        self.device_combo.currentIndexChanged.connect(self._device_changed)
        self.demo_check.stateChanged.connect(self._demo_toggled)
        self.half_check.stateChanged.connect(self._demo_options_changed)
        self.pink_check.stateChanged.connect(self._demo_options_changed)
        self.onehalf_check.stateChanged.connect(self._demo_options_changed)
        self.brown_check.stateChanged.connect(self._demo_options_changed)
        self.white_check.stateChanged.connect(self._demo_options_changed)
        self.transient_check.stateChanged.connect(self._demo_options_changed)
        self.avg_combo.currentIndexChanged.connect(self._avg_changed)
        return group

    def _build_calibration_group(self):
        group = QGroupBox("校准")
        layout = QVBoxLayout(group)
        layout.setSpacing(5)
        row = QHBoxLayout()
        self.cal_slider = QSlider(Qt.Horizontal)
        self.cal_slider.setRange(-30, 30)
        self.cal_slider.setValue(0)
        self.cal_label = QLabel("0 dB")
        self.cal_label.setFixedWidth(54)
        self.cal_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.cal_slider, 1)
        row.addWidget(self.cal_label)
        layout.addLayout(row)

        gate_row = QHBoxLayout()
        gate_row.addWidget(QLabel("静音门限"))
        self.gate_slider = QSlider(Qt.Horizontal)
        self.gate_slider.setRange(-100, -40)
        self.gate_slider.setValue(-80)
        self.gate_label = QLabel("-80 dBFS")
        self.gate_label.setFixedWidth(64)
        self.gate_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        gate_row.addWidget(self.gate_slider, 1)
        gate_row.addWidget(self.gate_label)
        layout.addLayout(gate_row)

        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("输入增益"))
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(-20, 30)
        self.gain_slider.setValue(0)
        self.gain_label = QLabel("0 dB")
        self.gain_label.setFixedWidth(54)
        self.gain_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        gain_row.addWidget(self.gain_slider, 1)
        gain_row.addWidget(self.gain_label)
        layout.addLayout(gain_row)

        denoise_row = QHBoxLayout()
        self.denoise_check = QCheckBox("去底噪")
        self.denoise_check.setChecked(True)
        self.denoise_slider = QSlider(Qt.Horizontal)
        self.denoise_slider.setRange(0, 100)
        self.denoise_slider.setValue(80)
        self.denoise_label = QLabel("80%")
        self.denoise_label.setFixedWidth(44)
        self.denoise_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        denoise_row.addWidget(self.denoise_check)
        denoise_row.addWidget(self.denoise_slider, 1)
        denoise_row.addWidget(self.denoise_label)
        layout.addLayout(denoise_row)

        note = QLabel("未校准麦克风；数字满幅按 94 dB 估算\n输入增益为数字增益；低于静音门限时清空实时显示")
        note.setWordWrap(True)
        note.setStyleSheet("color:#888;")
        layout.addWidget(note)
        self.cal_slider.valueChanged.connect(self._calibration_changed)
        self.gate_slider.valueChanged.connect(self._gate_changed)
        self.gain_slider.valueChanged.connect(self._gain_changed)
        self.denoise_check.stateChanged.connect(self._denoise_changed)
        self.denoise_slider.valueChanged.connect(self._denoise_changed)
        return group

    def _build_stats_group(self):
        group = QGroupBox("实时统计")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        self.current_label = QLabel("--.- dB(A)")
        self.current_label.setStyleSheet("font-size:27px; font-weight:600;")
        layout.addWidget(self.current_label)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(3)
        self.stat_labels = {}
        for key, text in (
            ("leq10", "10 s Leq"),
            ("l10", "L10"),
            ("l50", "L50"),
            ("l90", "L90"),
            ("peak", "峰值"),
            ("crest", "波峰因数"),
            ("dominant", "主频"),
            ("alpha", "1/f 斜率 α"),
            ("colour", "噪声类型"),
        ):
            value = QLabel("–")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.addRow(text, value)
            self.stat_labels[key] = value
        layout.addLayout(form)
        return group

    def _build_history_group(self):
        group = QGroupBox("长期记录")
        layout = QVBoxLayout(group)
        layout.setSpacing(4)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(3)
        self.hist_labels = {}
        for key, text in (
            ("count", "样本数"),
            ("avg24", "24 h 均值"),
            ("avg7", "7 d 均值"),
            ("max7", "7 d 最高"),
            ("alpha7", "7 d α"),
        ):
            value = QLabel("–")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            form.addRow(text, value)
            self.hist_labels[key] = value
        layout.addLayout(form)
        self.export_btn = QPushButton("导出 CSV")
        layout.addWidget(self.export_btn)
        self.export_btn.clicked.connect(self._export_csv)
        return group

    def _build_axes(self):
        self.ax_wave = self.figure.add_subplot(3, 2, 1)
        self.ax_spec = self.figure.add_subplot(3, 2, 2)
        self.ax_level = self.figure.add_subplot(3, 2, 3)
        self.ax_band = self.figure.add_subplot(3, 2, 4)
        self.ax_hist = self.figure.add_subplot(3, 2, 5)
        self.ax_hourly = self.figure.add_subplot(3, 2, 6)

        self.line_wave, = self.ax_wave.plot([], [], lw=0.8)
        self.ax_wave.set_title("时域波形（最近 93 ms）")
        self.ax_wave.set_xlim(0, BLOCK_SIZE)
        self.ax_wave.set_ylim(-0.1, 0.1)
        self.ax_wave.set_xlabel("采样点")

        self.line_spec, = self.ax_spec.plot([], [], lw=0.9)
        self.ax_spec.set_title("A 计权频谱 · 1/f 斜率")
        self.ax_spec.set_xscale("log")
        self.ax_spec.set_xlim(20, 20000)
        self.ax_spec.set_ylim(0, 70)
        self.ax_spec.set_xlabel("频率 (Hz)")
        self.ax_spec.set_ylabel("dB")
        self.fit_line, = self.ax_spec.plot([], [], lw=1.0, ls="--", color="#c0504d")
        self.ref_line, = self.ax_spec.plot([], [], lw=0.9, ls=":", color="#555")
        self.slope_text = self.ax_spec.text(
            0.02, 0.96, "", transform=self.ax_spec.transAxes,
            va="top", ha="left", fontsize=9,
        )

        self.line_level, = self.ax_level.plot([], [], lw=0.9)
        self.ax_level.set_title("A 计权声级（最近 60 s）")
        self.ax_level.set_xlim(0, 60)
        self.ax_level.set_ylim(20, 100)
        self.ax_level.set_xlabel("秒")
        self.ax_level.set_ylabel("dB(A)")

        self.band_line, = self.ax_band.plot(
            [], [], "o-", lw=1.0, ms=3.0, color="#4f7894",
        )
        self.ax_band.set_xlim(0, 50)
        self.ax_band.set_ylim(0, 1)
        self.ax_band.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        self._update_band_title()
        self.ax_band.set_xlabel("1/f (ms，线性坐标)")
        self.ax_band.set_ylabel("未计权能量密度 (FS^2/Hz)")
        self.band_peak_text = self.ax_band.text(
            0.02, 0.96, "", transform=self.ax_band.transAxes,
            va="top", ha="left", fontsize=9,
        )

        self.hist_bins = np.linspace(25, 95, 71)
        self.hist_centers = 0.5 * (self.hist_bins[:-1] + self.hist_bins[1:])
        self.hist_bars = self.ax_hist.bar(
            self.hist_centers, np.zeros(len(self.hist_centers)),
            width=(self.hist_bins[1] - self.hist_bins[0]) * 0.9,
            color="#777", alpha=0.75,
        )
        self.ax_hist.set_title("声级分布（最近 60 s）")
        self.ax_hist.set_xlim(25, 95)
        self.ax_hist.set_xlabel("dB(A)")
        self.ax_hist.set_ylabel("次数")
        self.hist_lines = [
            self.ax_hist.axvline(0, color=color, lw=0.8, ls="--")
            for color in ("#444", "#666", "#888")
        ]

        self.hour_bars = self.ax_hourly.bar(
            np.arange(24), np.zeros(24), color="#888", alpha=0.8
        )
        self.ax_hourly.set_title("长期规律：小时均值（7 天）")
        self.ax_hourly.set_xlim(-0.5, 23.5)
        self.ax_hourly.set_ylim(20, 95)
        self.ax_hourly.set_xlabel("小时")
        self.ax_hourly.set_ylabel("dB(A)")
        self.ax_hourly.set_xticks([0, 6, 12, 18, 23])

    # --------------------------------------------------------------- devices
    def _populate_devices(self):
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        try:
            devices = sd.query_devices()
        except Exception as exc:
            devices = []
            self.statusBar().showMessage(f"无法枚举音频设备：{exc}")
        for index, dev in enumerate(devices):
            if int(dev.get("max_input_channels", 0)) > 0:
                self.device_combo.addItem(f"{index}: {dev['name']}", index)
        if self.device_combo.count() == 0:
            self.device_combo.addItem("未检测到输入设备（演示模式）", None)
        self.device_combo.blockSignals(False)

    def _refresh_devices(self):
        self._stop_sources()
        self._populate_devices()
        if self.running:
            self._start_monitor()

    def _device_changed(self, _index):
        if self.running:
            self._stop_sources()
            self._start_monitor()

    # --------------------------------------------------------------- monitor
    def _demo_toggled(self, _state):
        if not self.running:
            return
        self._stop_sources()
        if self.demo_check.isChecked():
            self._start_demo("手动演示模式")
        else:
            self._start_monitor()

    def _start_monitor(self):
        self._stop_sources()
        if self.force_demo or self.demo_check.isChecked():
            self._start_demo("演示模式")
            return
        device = self.device_combo.currentData()
        if device is None:
            self._start_demo("未检测到输入设备，使用演示数据")
            return
        try:
            self.capture = AudioCapture(self.fs, BLOCK_SIZE, device=device)
            self.capture.start()
            self.demo = False
            self.noise_psd = None
            self.mode_label.setText(self.device_combo.currentText())
            self.statusBar().showMessage("正在采集环境噪声…")
        except Exception as exc:
            self.capture = None
            self._start_demo(f"打开麦克风失败：{exc}")

    def _start_demo(self, message):
        self.demo = True
        self.capture = None
        self.noise_psd = None
        self.demo_source = DemoSource(
            self.fs,
            include_half=self.half_check.isChecked(),
            include_pink=self.pink_check.isChecked(),
            include_one_half=self.onehalf_check.isChecked(),
            include_brown=self.brown_check.isChecked(),
            include_white=self.white_check.isChecked(),
            include_transient=self.transient_check.isChecked(),
        )
        self.mode_label.setText("演示数据（未校准）")
        self.statusBar().showMessage(message + "；不写入长期历史")
        self._update_long_term(force=True)

    def _stop_sources(self):
        if self.capture is not None:
            self.capture.stop()
            self.capture = None

    def _toggle_running(self):
        self.running = not self.running
        if self.running:
            self.start_btn.setText("暂停")
            self._start_monitor()
            self.session_started = time.time()
        else:
            self.start_btn.setText("开始")
            self._stop_sources()
            self.statusBar().showMessage("已暂停")

    def _clear_live(self, message=""):
        self.levels.clear()
        self.density_history.clear()
        self.noise_psd = None
        self.line_wave.set_data([], [])
        self.line_spec.set_data([], [])
        self.line_level.set_data([], [])
        self.fit_line.set_data([], [])
        self.ref_line.set_data([], [])
        self.slope_text.set_text("")
        self.band_line.set_data([], [])
        self.band_peak_text.set_text("")
        for bar in self.hist_bars:
            bar.set_height(0.0)
        self.current_label.setText("--.- dB(A)")
        for label in self.stat_labels.values():
            label.setText("–")
        if message:
            self.statusBar().showMessage(message)
        self.canvas.draw_idle()

    def _demo_options_changed(self, _state):
        if not self.demo:
            return
        self.demo_source = DemoSource(
            self.fs,
            include_half=self.half_check.isChecked(),
            include_pink=self.pink_check.isChecked(),
            include_one_half=self.onehalf_check.isChecked(),
            include_brown=self.brown_check.isChecked(),
            include_white=self.white_check.isChecked(),
            include_transient=self.transient_check.isChecked(),
        )
        self.density_history.clear()
        self.band_line.set_data([], [])
        self.band_peak_text.set_text("")
        if not self.demo_source.enabled:
            self._clear_live("未选择演示噪声成分")

    def _calibration_changed(self, value):
        self.calibration = float(value)
        self.cal_label.setText(f"{value:+d} dB" if value else "0 dB")

    def _denoise_changed(self, _value=None):
        self.denoise_strength = self.denoise_slider.value() / 100.0
        self.denoise_label.setText(f"{self.denoise_slider.value()}%")
        self.noise_psd = None

    def _gate_changed(self, value):
        self.silence_gate_db = float(value)
        self.gate_label.setText(f"{value} dBFS")

    def _gain_changed(self, value):
        self.input_gain_db = float(value)
        self.gain_label.setText(f"{value:+d} dB" if value else "0 dB")

    def _avg_changed(self, _index):
        self.avg_seconds = int(self.avg_combo.currentData() or 10)
        self.density_history.clear()
        self.band_line.set_data([], [])
        self.band_peak_text.set_text("")
        self._update_band_title()
        self.canvas.draw_idle()

    def _update_band_title(self):
        self.ax_band.set_title(
            f"1/频率-未计权能量密度（{self.avg_seconds} s 时间平均）"
        )

    def _reset_session(self):
        self.levels.clear()
        self.density_history.clear()
        self.noise_psd = None
        self.band_line.set_data([], [])
        self.band_peak_text.set_text("")
        self.session_samples = 0
        self.session_started = time.time()
        for label in self.stat_labels.values():
            label.setText("–")
        self.current_label.setText("--.- dB(A)")
        self.statusBar().showMessage("本次会话统计已清空")

    # ------------------------------------------------------------------ data
    def _on_tick(self):
        if not self.running:
            return
        if self.demo:
            if not self.demo_source.enabled:
                self._clear_live("未选择演示噪声成分")
                return
            block = self.demo_source.block(int(self.fs / DISPLAY_HZ))
        else:
            if self.capture is None:
                return
            block = self.capture.drain()
            if block is None:
                return
        self._process(block)
        self._draw_counter += 1
        if self._draw_counter % DRAW_EVERY == 0:
            self.canvas.draw_idle()

    def _process(self, block):
        block = np.asarray(block, dtype=np.float64).reshape(-1)
        if len(block) < 256:
            return
        if self.input_gain_db:
            block = block * (10.0 ** (self.input_gain_db / 20.0))
        peak = float(np.max(np.abs(block)))
        analysis = block[-min(len(block), BLOCK_SIZE):]
        rms = float(np.sqrt(np.mean(analysis * analysis)))
        rms_dbfs = 20.0 * np.log10(max(rms, EPS))
        if rms_dbfs < self.silence_gate_db:
            self._clear_live(
                f"低于静音门限 {self.silence_gate_db:.0f} dBFS，清空实时显示"
            )
            return
        freqs, power = spectral_power(analysis, self.fs)
        if self.denoise_check.isChecked():
            if self.noise_psd is None or len(self.noise_psd) != len(power):
                self.noise_psd = power.copy()
            else:
                down = power < self.noise_psd
                self.noise_psd[down] = 0.65 * self.noise_psd[down] + 0.35 * power[down]
                self.noise_psd[~down] = 0.995 * self.noise_psd[~down] + 0.005 * power[~down]
            power = subtract_noise(power, self.noise_psd, self.denoise_strength)
        spec_db, z_db, a_db = analyse_power(freqs, power, self.calibration)
        self.levels.append(a_db)
        self.session_samples += 1

        visible = analysis
        self.line_wave.set_data(np.arange(len(visible)), visible)
        y_max = max(0.002, float(np.max(np.abs(visible))) * 1.25)
        self.ax_wave.set_ylim(-y_max, y_max)

        self.line_spec.set_data(freqs, spec_db)
        raw_db = spec_db - a_weighting_db(freqs)
        alpha, fit_slope, fit_intercept, _r2 = spectral_slope(freqs, raw_db)
        if np.isfinite(alpha):
            fit_db = fit_curve_db(freqs, fit_slope, fit_intercept) + a_weighting_db(freqs)
            ref_db = reference_1f_db(freqs, raw_db) + a_weighting_db(freqs)
            self.fit_line.set_data(freqs, fit_db)
            self.ref_line.set_data(freqs, ref_db)
            self.slope_text.set_text(
                f"α = {alpha:.2f}  {noise_colour(alpha)}" + chr(10)
                + "虚线：拟合    点线：1/f 参考"
            )
        else:
            self.fit_line.set_data([], [])
            self.ref_line.set_data([], [])
            self.slope_text.set_text("")
        # The 1/f law applies to the unweighted PSD S(f), not the
        # A-weighted display spectrum.  Fit this panel on raw_db.
        _, density = third_octave_density(
            freqs, raw_db, self.band_centers, self.calibration
        )
        now_band = time.time()
        self.density_history.append((now_band, density.copy()))
        cutoff = now_band - self.avg_seconds
        while self.density_history and self.density_history[0][0] < cutoff:
            self.density_history.popleft()
        stack = np.vstack([item[1] for item in self.density_history])
        valid = np.isfinite(stack)
        counts = valid.sum(axis=0)
        sums = np.where(valid, stack, 0.0).sum(axis=0)
        density = np.divide(
            sums, counts,
            out=np.full(self.band_centers.shape, np.nan, dtype=np.float64),
            where=counts > 0,
        )
        x_inv = 1000.0 / self.band_centers
        density_plot = density
        order = np.argsort(x_inv)
        self.band_line.set_data(x_inv[order], density_plot[order])
        if np.any(np.isfinite(density_plot)):
            peak_i = int(np.nanargmax(density_plot))
            self.band_peak_text.set_text(
                f"峰值密度 {self.band_centers[peak_i]:.0f} Hz · "
                f"{density_plot[peak_i]:.2e} FS^2/Hz"
            )
            self.ax_band.set_ylim(
                0.0, max(float(np.nanmax(density_plot)) * 1.18, 1e-12)
            )
        else:
            self.band_line.set_data([], [])
            self.band_peak_text.set_text("")

        values = np.asarray(self.levels, dtype=np.float64)
        self.line_level.set_data(np.arange(len(values)) / DISPLAY_HZ, values)
        self.ax_level.set_xlim(0, 60)
        if len(values) > 1:
            self.ax_level.set_ylim(
                max(20.0, float(np.nanmin(values)) - 5.0),
                min(110.0, float(np.nanmax(values)) + 5.0),
            )
        leq10 = leq(values[-DISPLAY_HZ * 10:])
        l10, l50, l90 = percentile_levels(values[-DISPLAY_HZ * 60:], (10, 50, 90))
        hist, _ = np.histogram(values[-DISPLAY_HZ * 60:], bins=self.hist_bins)
        for bar, height in zip(self.hist_bars, hist):
            bar.set_height(float(height))
        for line, value in zip(self.hist_lines, (l10, l50, l90)):
            line.set_xdata([value, value])
        self.ax_hist.set_ylim(0, max(5.0, float(hist.max()) * 1.15 if len(hist) else 5.0))

        peak_db = _spl_from_rms(peak, self.calibration)
        crest = peak / max(rms, EPS)
        dominant = float(freqs[int(np.argmax(spec_db))]) if len(freqs) else 0.0
        self.current_label.setText(f"{a_db:5.1f} dB(A)")
        self.stat_labels["leq10"].setText(f"{leq10:5.1f} dB")
        self.stat_labels["l10"].setText(f"{l10:5.1f} dB")
        self.stat_labels["l50"].setText(f"{l50:5.1f} dB")
        self.stat_labels["l90"].setText(f"{l90:5.1f} dB")
        self.stat_labels["peak"].setText(f"{peak_db:5.1f} dB")
        self.stat_labels["crest"].setText(f"{crest:4.1f}")
        self.stat_labels["dominant"].setText(f"{dominant:5.0f} Hz")
        self.stat_labels["alpha"].setText(
            "–" if not np.isfinite(alpha) else f"{alpha:4.2f}"
        )
        self.stat_labels["colour"].setText(noise_colour(alpha))

        now = time.time()
        if not self.demo and a_db > 10.0 and now - self.last_save >= 1.0:
            self.last_save = now
            self.history.add(now, z_db, a_db, peak_db, alpha)
        if now - self.last_long_update >= 20.0:
            self._update_long_term()

    # -------------------------------------------------------------- long term
    def _update_long_term(self, force=False):
        now = time.time()
        self.last_long_update = now
        if self.demo:
            hours = np.arange(24)
            profile = 52.0 + 6.0 * np.sin((hours - 7) / 24 * 2 * np.pi)
            profile += 4.0 * np.exp(-((hours - 8) ** 2) / 8.0)
            profile += 5.0 * np.exp(-((hours - 18) ** 2) / 10.0)
            self._set_hourly(profile)
            self.hist_labels["count"].setText("演示")
            self.hist_labels["avg24"].setText("–")
            self.hist_labels["avg7"].setText("–")
            self.hist_labels["max7"].setText("–")
            self.hist_labels["alpha7"].setText("–")
            return

        summary = self.history.summary(7)
        profile = self.history.hourly_profile(7)
        self._set_hourly(profile)
        self.hist_labels["count"].setText(f"{summary['count']:,}")
        self.hist_labels["avg24"].setText(
            "–" if not np.isfinite(self.history.summary(1)["avg"])
            else f"{self.history.summary(1)['avg']:.1f} dB"
        )
        self.hist_labels["avg7"].setText(
            "–" if not np.isfinite(summary["avg"]) else f"{summary['avg']:.1f} dB"
        )
        self.hist_labels["max7"].setText(
            "–" if not np.isfinite(summary["peak"]) else f"{summary['peak']:.1f} dB"
        )
        self.hist_labels["alpha7"].setText(
            "–" if not np.isfinite(summary["alpha"]) else f"{summary['alpha']:.2f}"
        )

    def _set_hourly(self, profile):
        values = np.asarray(profile, dtype=np.float64)
        for bar, value in zip(self.hour_bars, values):
            bar.set_height(0.0 if not np.isfinite(value) else float(value))
        if np.any(np.isfinite(values)):
            low = max(20.0, float(np.nanmin(values)) - 8.0)
            high = min(110.0, float(np.nanmax(values)) + 8.0)
            self.ax_hourly.set_ylim(low, high)
        self.ax_hourly.set_title("长期规律：小时均值（7 天）")

    # ----------------------------------------------------------------- export
    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出噪声历史", str(ROOT / "noise_history.csv"), "CSV (*.csv)"
        )
        if not path:
            return
        try:
            count = self.history.export_csv(Path(path))
            QMessageBox.information(self, "导出完成", f"已导出 {count} 条记录")
        except Exception as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def closeEvent(self, event):
        self.timer.stop()
        self._stop_sources()
        self.history.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = NoiseWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
