from pathlib import Path

import numpy as np
import pytest
import reportlab
from pypdf import PdfReader

from pylex.cli import main
from pylex.config import Config
from pylex.overlay import build_invisible_text_overlay, resolve_font

POLY = np.array([[10, 10], [210, 10], [210, 40], [10, 40]], dtype=float)
VERA = Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"  # ships with reportlab


def extract(text, font="Helvetica"):
    buf = build_invisible_text_overlay([(text, POLY, 0.9)], (400, 300), 400, 300, 400, 300, 0, 0, 0, 0, Config(font=font))
    return PdfReader(buf).pages[0].extract_text()


def test_latin_text_round_trips_with_the_default_font():
    assert extract("caf\u00e9 Z\u00fcrich 123") == "caf\u00e9 Z\u00fcrich 123"


def test_ttf_path_is_registered_and_used():
    assert resolve_font(str(VERA)) == "pylex-Vera"
    assert extract("caf\u00e9 123", font=str(VERA)) == "caf\u00e9 123"


def test_builtin_cjk_font_round_trips_chinese():
    assert extract("\u4f60\u597d\u4e16\u754c", font="STSong-Light") == "\u4f60\u597d\u4e16\u754c"


def test_non_latin_text_with_the_default_font_warns(caplog):
    with caplog.at_level("WARNING", logger="pylex"):
        extract("\u041f\u0440\u0438\u0432\u0435\u0442")

    assert "--font" in caplog.text


def test_unknown_font_is_rejected_with_guidance():
    with pytest.raises(ValueError, match="Unknown font"):
        resolve_font("NoSuchFont")


def test_cli_rejects_a_bad_font_up_front(capsys):
    with pytest.raises(SystemExit) as info:
        main(["a.png", "--font", "NoSuchFont"])

    assert info.value.code == 2
    assert "Unknown font" in capsys.readouterr().err
