"""Small dependency-free progress display for the command-line runner.

Renders an in-place updating bar on a real terminal and lets callers print
normal status lines via println() without corrupting it. Falls back to
plain per-item lines (no carriage-return / bar noise) when stdout isn't a
terminal — piped output, redirected to a log file, or --quiet — so a
redirected run's log stays clean.

The startup spinner is deliberately owned by this class rather than by the
OCR backend. PaddleOCR loads in child processes on Windows, so the parent
cannot rely on backend log output to show that work is progressing. The
spinner keeps the terminal (and redirected logs) informative while workers
are loading their private engines.

No external dependency (tqdm etc.) on purpose: this tool's own progress
needs are simple (one bar over a batch of files, an optional live suffix
for per-page progress within one big PDF), and pulling in a dependency
just for that isn't worth it.
"""

from __future__ import annotations

import shutil
import sys
import threading

_SPINNER_FRAMES = ("|", "/", "-", "\\")


def _terminal_columns() -> int:
    return shutil.get_terminal_size(fallback=(100, 24)).columns


class ProgressBar:
    def __init__(self, total: int, label: str = "", enabled: bool = True):
        self.total = max(int(total), 0)
        self.label = label
        self.count = 0
        self._progress = 0.0
        self._suffix = ""
        self._enabled = bool(enabled)
        self._is_tty = bool(enabled and sys.stdout.isatty() and self.total > 0)
        self._lock = threading.RLock()
        self._spinner_stop: threading.Event | None = None
        self._spinner_thread: threading.Thread | None = None

    def _render(self) -> None:
        if not self._is_tty:
            return

        frac = (self._progress / self.total) if self.total else 1.0
        prefix = f"{self.label} " if self.label else ""
        if abs(self._progress - round(self._progress)) < 1e-9:
            progress_count = str(round(self._progress))
        else:
            progress_count = f"{self._progress:.2f}".rstrip("0").rstrip(".")
        stats = f" {int(frac * 100):3d}% ({progress_count}/{self.total})"
        suffix = f" {self._suffix}" if self._suffix else ""

        cols = _terminal_columns()
        reserved = len(prefix) + len(stats) + len(suffix) + 4  # brackets + padding
        width = max(10, min(40, cols - reserved))

        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        line = f"{prefix}[{bar}]{stats}{suffix}"

        sys.stdout.write("\r" + line[:cols].ljust(cols))
        sys.stdout.flush()

    def start_spinner(self, text: str, interval: float = 0.2) -> None:
        """Show activity while work is running without advancing the bar.

        On a real terminal this animates in place. For redirected output it
        emits one plain line, which gives users evidence that engine loading
        has started without introducing carriage-return noise. ``--quiet``
        remains completely silent.
        """
        if not self._enabled:
            return

        self.stop_spinner()
        with self._lock:
            self._suffix = f"{text} {_SPINNER_FRAMES[0]}"
            if not self._is_tty:
                print(text)
                return
            self._render()
            stop = threading.Event()
            self._spinner_stop = stop

        def _spin() -> None:
            frame_index = 0
            while not stop.wait(interval):
                frame_index = (frame_index + 1) % len(_SPINNER_FRAMES)
                with self._lock:
                    if self._spinner_stop is not stop:
                        return
                    self._suffix = f"{text} {_SPINNER_FRAMES[frame_index]}"
                    self._render()

        thread = threading.Thread(target=_spin, name="pidocr-progress-spinner", daemon=True)
        with self._lock:
            self._spinner_thread = thread
        thread.start()

    def stop_spinner(self) -> None:
        """Stop a startup spinner, leaving the bar ready for normal updates."""
        with self._lock:
            stop = self._spinner_stop
            thread = self._spinner_thread
            self._spinner_stop = None
            self._spinner_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        with self._lock:
            self._suffix = ""

    def set_suffix(self, text: str) -> None:
        """Update the trailing status text without advancing the count."""
        with self._lock:
            self._suffix = text
            self._render()

    def update(self, n: int = 1) -> None:
        self.stop_spinner()
        with self._lock:
            self.count = min(self.count + n, self.total)
            self._progress = float(self.count)
            self._suffix = ""
            self._render()

    def set_progress(self, value: float, suffix: str | None = None) -> None:
        """Render fractional progress without marking a file complete.

        ``value`` is measured in completed-file units. This lets callers show
        page/tile work for a single input while retaining the familiar
        ``completed/total`` display when a file finishes.
        """
        if not self._enabled:
            return
        with self._lock:
            requested = max(0.0, min(float(value), float(self.total)))
            self._progress = max(self._progress, requested)
            if suffix is not None:
                self._suffix = suffix
            self._render()

    def println(self, text: str = "") -> None:
        """Print a normal status line without corrupting the in-place bar.
        On a non-terminal this is just print() — no bar to protect. When
        the bar was constructed with enabled=False (e.g. --quiet), this is a
        no-op: quiet means quiet, not just "no animated bar".
        """
        if not self._enabled:
            return
        with self._lock:
            if self._is_tty:
                sys.stdout.write("\r" + " " * _terminal_columns() + "\r")
                sys.stdout.flush()
            if text:
                print(text)
            if self._is_tty:
                self._render()

    def close(self) -> None:
        self.stop_spinner()
        with self._lock:
            if self._is_tty:
                sys.stdout.write("\n")
                sys.stdout.flush()
