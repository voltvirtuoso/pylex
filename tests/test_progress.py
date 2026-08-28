from pylex.progress import ProgressBar


def test_progress_bar_disabled_on_non_tty(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    bar = ProgressBar(total=5, label="OCR")
    assert bar._is_tty is False

    bar.update(1)
    bar.println("hello")
    captured = capsys.readouterr()
    assert "hello" in captured.out
    # No carriage-return bar noise when not a tty.
    assert "\r" not in captured.out


def test_progress_bar_enabled_false_never_renders(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    bar = ProgressBar(total=5, label="OCR", enabled=False)
    assert bar._is_tty is False

    bar.update(1)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_progress_bar_enabled_false_println_suppressed(monkeypatch, capsys):
    """--quiet should mean quiet: println must not fall back to a plain
    print() just because there's no animated bar to protect."""
    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    bar = ProgressBar(total=5, label="OCR", enabled=False)

    bar.println("this should not appear")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_progress_bar_tty_renders_bar(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    bar = ProgressBar(total=4, label="OCR")
    assert bar._is_tty is True

    bar.update(2)
    captured = capsys.readouterr()
    assert "\r" in captured.out
    assert "2/4" in captured.out
    assert "50%" in captured.out


def test_progress_bar_count_never_exceeds_total(monkeypatch):
    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    bar = ProgressBar(total=3)
    bar.update(10)
    assert bar.count == 3


def test_progress_bar_zero_total_disables_tty_rendering(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    bar = ProgressBar(total=0)
    assert bar._is_tty is False
    bar.println("no-op bar, still prints lines")
    captured = capsys.readouterr()
    assert "no-op bar, still prints lines" in captured.out


def test_progress_bar_println_interleaves_without_crash(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    bar = ProgressBar(total=2, label="OCR")
    bar.update(1)
    bar.println("file 1 done")
    bar.update(1)
    bar.close()
    captured = capsys.readouterr()
    assert "file 1 done" in captured.out


def test_progress_bar_startup_status_prints_once_when_redirected(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    bar = ProgressBar(total=2, label="OCR")
    bar.start_spinner("Loading OCR engine", interval=0.01)
    bar.stop_spinner()
    captured = capsys.readouterr()
    assert captured.out == "Loading OCR engine\n"
    assert "\r" not in captured.out


def test_progress_bar_startup_status_respects_quiet(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    bar = ProgressBar(total=2, label="OCR", enabled=False)
    bar.start_spinner("Loading OCR engine", interval=0.01)
    bar.stop_spinner()
    captured = capsys.readouterr()
    assert captured.out == ""


def test_progress_bar_fractional_progress_renders_intermediate_state(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    bar = ProgressBar(total=1, label="OCR")
    bar.set_progress(0.5, suffix="image.png: OCR tile 1/2")

    captured = capsys.readouterr()
    assert "50%" in captured.out
    assert "0.5/1" in captured.out
    assert "OCR tile 1/2" in captured.out


def test_progress_bar_fractional_progress_never_moves_backward(monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    bar = ProgressBar(total=1, label="OCR")
    bar.set_progress(0.75, suffix="OCR complete")
    bar.set_progress(0.05, suffix="Building searchable PDF")

    captured = capsys.readouterr()
    assert bar._progress == 0.75
    assert "75%" in captured.out
