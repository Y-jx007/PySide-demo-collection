import sys
import numpy as np
import sounddevice as sd
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QStatusBar)
from PySide6.QtCore import QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# ---------- 修复中文显示 ----------
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'PingFang SC',
    'Noto Sans CJK SC', 'WenQuanYi Micro Hei'
]
plt.rcParams['axes.unicode_minus'] = False
# ---------------------------------

# ---------- 音频参数 ----------
SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
BUFFER_SECONDS = 2
RING_SIZE = SAMPLE_RATE * BUFFER_SECONDS

def a_weighting(freq):
    f2 = freq ** 2
    ra = (12200 ** 2 * f2 ** 2) / (
        (f2 + 20.6 ** 2) * (f2 + 12200 ** 2) *
        np.sqrt(f2 + 107.7 ** 2) * np.sqrt(f2 + 737.9 ** 2)
    )
    return 20 * np.log10(ra) + 2.0

FREQS = np.fft.rfftfreq(BLOCK_SIZE, 1 / SAMPLE_RATE)
A_WEIGHTS = a_weighting(FREQS)

ring_buffer = np.zeros(RING_SIZE, dtype=np.float32)
buffer_index = 0

def audio_callback(indata, frames, time, status):
    global ring_buffer, buffer_index
    if status:
        print(status)
    mono = indata[:, 0]
    n = len(mono)
    if buffer_index + n <= RING_SIZE:
        ring_buffer[buffer_index:buffer_index + n] = mono
        buffer_index += n
    else:
        rem = RING_SIZE - buffer_index
        ring_buffer[buffer_index:] = mono[:rem]
        ring_buffer[:n - rem] = mono[rem:]
        buffer_index = n - rem

# ---------- 主窗口 ----------
class NoiseAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("环境噪声实时统计 (PySide6)")
        self.resize(1000, 700)

        self.stream = None

        self.figure = Figure(figsize=(10, 7), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.canvas)

        self.statusBar().showMessage("正在采集环境噪声...")

        self.ax_wave = self.figure.add_subplot(3, 2, 1)
        self.ax_spec = self.figure.add_subplot(3, 2, 2)
        self.ax_level = self.figure.add_subplot(3, 2, 3)
        self.ax_hist = self.figure.add_subplot(3, 2, 4)
        self.ax_text = self.figure.add_subplot(3, 2, 5)
        self.ax_spectrogram = self.figure.add_subplot(3, 2, 6)

        self.line_wave, = self.ax_wave.plot([], [], lw=1)
        self.ax_wave.set_title("时域波形 (最近 200 ms)")
        self.ax_wave.set_xlim(0, BLOCK_SIZE)
        self.ax_wave.set_ylim(-1, 1)

        self.line_spec, = self.ax_spec.plot([], [], lw=1)
        self.ax_spec.set_title("功率谱 (A 计权)")
        self.ax_spec.set_xscale('log')
        self.ax_spec.set_xlim(20, SAMPLE_RATE // 2)
        self.ax_spec.set_ylim(-80, 20)
        self.ax_spec.set_xlabel("频率 (Hz)")
        self.ax_spec.set_ylabel("dB")

        self.line_level, = self.ax_level.plot([], [], lw=1)
        self.ax_level.set_title("A 计权声压级 (dB SPL)")
        self.ax_level.set_xlim(0, 600)
        self.ax_level.set_ylim(20, 90)
        self.ax_level.set_xlabel("秒/10")
        self.ax_level.set_ylabel("dB")

        self.ax_hist.set_title("幅值分布 (最近 2 秒)")
        self.hist_bins = np.linspace(-1, 1, 100)

        self.ax_text.axis('off')
        self.stats_text = self.ax_text.text(0.05, 0.5, "", fontsize=10, va='center')

        self.img = self.ax_spectrogram.imshow(
            np.zeros((50, len(FREQS))),
            aspect='auto', origin='lower', cmap='inferno',
            extent=[20, SAMPLE_RATE // 2, 0, 5]
        )
        self.ax_spectrogram.set_title("频谱时间变化")
        self.ax_spectrogram.set_xscale('log')
        self.ax_spectrogram.set_xlabel("频率 (Hz)")
        self.ax_spectrogram.set_ylabel("秒")

        self.level_history = []
        self.spec_history = np.zeros((50, len(FREQS)))

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)

        self.start_audio()

    def start_audio(self):
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            callback=audio_callback,
            dtype='float32'
        )
        self.stream.start()

    def update_plots(self):
        global ring_buffer, buffer_index
        if buffer_index < BLOCK_SIZE:
            return

        if buffer_index >= BLOCK_SIZE:
            data = ring_buffer[buffer_index - BLOCK_SIZE:buffer_index].copy()
        else:
            data = np.concatenate([ring_buffer[buffer_index - BLOCK_SIZE:],
                                   ring_buffer[:buffer_index]])

        data = data - np.mean(data)

        self.line_wave.set_data(np.arange(len(data)), data)
        y_max = max(0.01, np.abs(data).max() * 1.2)
        self.ax_wave.set_ylim(-y_max, y_max)

        windowed = data * np.hanning(BLOCK_SIZE)
        fft = np.fft.rfft(windowed)
        mag_db = 20 * np.log10(np.abs(fft) + 1e-10) + A_WEIGHTS
        self.line_spec.set_data(FREQS, mag_db)

        rms = np.sqrt(np.mean(data ** 2))
        dbspl = 94 + 20 * np.log10(rms + 1e-10) if rms > 0 else 0
        self.level_history.append(dbspl)
        if len(self.level_history) > 600:
            self.level_history = self.level_history[-600:]
        if self.level_history:
            self.line_level.set_data(range(len(self.level_history)), self.level_history)
            self.ax_level.set_xlim(max(0, len(self.level_history) - 600),
                                   len(self.level_history))

        self.ax_hist.clear()
        self.ax_hist.set_title("幅值分布 (最近 2 秒)")
        self.ax_hist.hist(ring_buffer, bins=self.hist_bins, color='gray', alpha=0.7)
        self.ax_hist.set_xlim(-1, 1)

        peak = np.max(np.abs(data))
        crest = peak / (rms + 1e-10)
        stats = (f"RMS: {rms:.4f}\n"
                 f"峰值: {peak:.4f}\n"
                 f"估计 SPL: {dbspl:.1f} dB\n"
                 f"波峰因数: {crest:.1f}\n"
                 f"采样率: {SAMPLE_RATE} Hz")
        self.stats_text.set_text(stats)

        self.spec_history = np.roll(self.spec_history, -1, axis=0)
        self.spec_history[-1, :] = mag_db
        self.img.set_data(self.spec_history)
        self.img.set_clim(-60, 10)

        self.canvas.draw_idle()

    def closeEvent(self, event):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NoiseAnalyzer()
    window.show()
    sys.exit(app.exec())