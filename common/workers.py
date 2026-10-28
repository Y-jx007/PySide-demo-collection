# -*- coding: utf-8 -*-
"""可协作停止的后台线程基类。

Arnold 的 CatMapWorker、Traveler 的 TSPWorker 原先各自维护一个
"_is_running / running" 布尔量，再各自写一遍 stop() 和
"stop -> wait -> terminate" 的收尾逻辑。统一到 StoppableThread 后：

    class MyWorker(StoppableThread):
        def run(self):
            while not self.stopping:
                ...

    self.worker = MyWorker(...)
    self.worker.start()
    ...
    self.worker.stop_and_wait(2000)
"""
from __future__ import annotations

from PySide6.QtCore import QThread


class StoppableThread(QThread):
    """带协作式停止标志的 QThread。

    子类在 run() 里定期检查 self.stopping；stop() 只置标志，
    不打断正在执行的这一步，避免 numpy/numba 循环里出现半截状态。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_requested = False

    # -- 停止标记 ----------------------------------------------------
    @property
    def stopping(self) -> bool:
        return self._stop_requested

    def stop(self) -> None:
        self._stop_requested = True

    def reset_stop(self) -> None:
        """复用同一个线程对象时清掉停止标记。"""
        self._stop_requested = False

    # -- 收尾 --------------------------------------------------------
    def stop_and_wait(self, timeout_ms: int = 3000) -> bool:
        """请求停止并等待；超时则强制 terminate。返回是否正常退出。"""
        self.stop()
        if not self.isRunning():
            return True
        if self.wait(timeout_ms):
            return True
        self.terminate()
        self.wait(1000)
        return False
