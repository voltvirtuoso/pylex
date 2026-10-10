from pathlib import Path

from pylex.pipeline import resolve_output_path


def test_several_inputs_with_one_output_name_get_separate_files(tmp_path):
    outputs = {
        resolve_output_path(Path(name), None, str(tmp_path / "out"), ".ocr", False, multiple_inputs=True)
        for name in ("a.png", "b.png")
    }

    assert outputs == {tmp_path / "out" / "a.pdf", tmp_path / "out" / "b.pdf"}


def test_single_input_keeps_the_exact_output_path_given():
    out = resolve_output_path(Path("a.png"), None, "result.pdf", ".ocr", False)

    assert out == Path("result.pdf")


def test_default_output_sits_next_to_the_input_with_the_suffix():
    assert resolve_output_path(Path("dir/scan.pdf"), None, None, ".ocr", False) == Path("dir/scan.ocr.pdf")


def test_file_result_tracks_pages_whose_ocr_failed():
    from pylex.pipeline import FileResult

    result = FileResult(input_path=Path("a.pdf"), ok=True)
    result.failed_pages.append(2)

    assert result.ok and result.failed_pages == [2]
    assert FileResult(input_path=Path("b.pdf")).failed_pages == []  # not shared between instances
