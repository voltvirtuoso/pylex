"""Every command and snippet shown in the README and docs must be valid against the real code."""

import ast
import re
import shlex
from pathlib import Path

import pytest

from pylex.cli import build_arg_parser

ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = [ROOT / "README.md", ROOT / "PERFORMANCE.md", *sorted((ROOT / "docs").glob("*.md"))]
FENCE = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)


def blocks(language):
    for path in DOC_FILES:
        for match in FENCE.finditer(path.read_text(encoding="utf-8")):
            if match.group(1) == language:
                yield path.name, match.group(2)


def pylex_commands():
    for name, body in blocks("bash"):
        for raw in body.replace("\\\n", " ").splitlines():  # join `\` line continuations
            line = raw.strip()
            if not line.startswith("pylex ") or line.startswith("pylex doctor") or "<" in line:
                continue
            yield name, line


@pytest.mark.parametrize(("doc", "command"), list(pylex_commands()))
def test_documented_pylex_command_is_accepted_by_the_real_parser(doc, command, capsys):
    argv = shlex.split(command, comments=True)[1:]  # drop "pylex", ignore trailing "# comments"
    try:
        build_arg_parser().parse_args(argv)
    except SystemExit as exc:  # argparse exits on invalid input; --help and --version exit 0
        if exc.code != 0:
            pytest.fail(f"{doc}: `{command}` is rejected: {capsys.readouterr().err.strip().splitlines()[-1]}")


@pytest.mark.parametrize(("doc", "code"), list(blocks("python")))
def test_documented_python_snippet_is_valid_syntax(doc, code):
    try:
        ast.parse(code)
    except SyntaxError as exc:
        pytest.fail(f"{doc}: invalid Python snippet ({exc.msg}, line {exc.lineno}):\n{code[:200]}")


def test_documented_doctor_options_exist(monkeypatch, capsys):
    from pylex.cli import main

    for _doc, body in blocks("bash"):
        for raw in body.splitlines():
            line = raw.strip()
            if line.startswith("pylex doctor"):
                options = [a for a in shlex.split(line, comments=True)[2:] if a.startswith("-")]
                for option in options:
                    assert option in {"--json", "--verify-gpu", "--install", "-y", "--yes"}, f"unknown doctor option {option}"
    assert main is not None


def test_relative_links_in_markdown_point_at_real_files():
    broken = []
    link = re.compile(r"\]\((?!https?:|#|mailto:)([^)\s#]+)")
    for path in DOC_FILES:
        for target in link.findall(path.read_text(encoding="utf-8")):
            if not (path.parent / target).resolve().exists():
                broken.append(f"{path.name} -> {target}")
    assert not broken, "broken relative links:\n" + "\n".join(broken)
