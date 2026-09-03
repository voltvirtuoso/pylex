# Pylex SEO and release implementation guide

This folder contains replacement and new project files. It does not modify the original repository checkout.

## Copy these files

Copy `README.md`, `pyproject.toml`, `CHANGELOG.md`, `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `CITATION.cff` to the repository root. Copy `docs/`, `.github/ISSUE_TEMPLATE/`, and `.github/workflows/` into the corresponding repository directories.

## Important naming decision

The supplied `pyproject.toml` uses the distribution name `pylex-ocr` because PyPI already has an unrelated project named `pylex`. The import remains `import pylex`, and the executable remains `pylex`. Confirm the name is available at publication time; PyPI registration can change.

If you obtain and intentionally use the exact `pylex` distribution name instead, change only the `name` field and the installation commands in the README and quickstart. Do not leave the stale `haroon/pylex` links in any file.

## Set GitHub repository metadata manually

In the repository About panel, set the description to:

> Python OCR and scanned-document text extraction CLI/API that creates searchable PDFs from images and PDFs, with batch processing, tiled OCR, and rotated-page-safe text placement.

Set the website to the canonical documentation URL or the repository URL until a documentation site exists. Add these topics: `python`, `ocr`, `optical-character-recognition`, `document-processing`, `pdf`, `searchable-pdf`, `pdf-ocr`, `text-extraction`, `batch-ocr`, `paddleocr`, `onnxruntime`, and `openvino`.

Enable Issues and Discussions if you are prepared to answer them. Enable Dependabot alerts, secret scanning, push protection, and code scanning where available for the public repository.

## Validate locally

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
python -m build
python -m twine check dist/*
```

Build and inspect the wheel before publishing. Confirm that the wheel contains the `pylex` package and console entry point, and that the rendered project description shows the intended README.

## Publish a release

After review, create the version tag and GitHub Release:

```bash
git add README.md pyproject.toml CHANGELOG.md LICENSE SECURITY.md CONTRIBUTING.md CODE_OF_CONDUCT.md CITATION.cff docs .github
 git commit -m "Improve documentation, metadata, and release readiness"
git tag -a v2.0.0 -m "Pylex 2.0.0"
git push origin main --follow-tags
```

Publish to PyPI only after deciding the distribution name and configuring trusted publishing or a protected API token. The package index name, GitHub repository URL, README installation command, and release name must all agree.

## Suggested GitHub Release text

```markdown
# Pylex 2.0.0

Pylex is a Python OCR CLI and library for creating searchable PDFs and extracting text regions from images and documents.

## Install

python -m pip install "pylex-ocr[ocr]"

## Highlights

- Searchable PDF output from images and scanned PDFs.
- Batch processing for files and folders.
- Tiled OCR for dense documents and small labels.
- Rotated-page-safe OCR text placement.
- Reusable Python API with confidence, polygons, and progress callbacks.

See the [README](https://github.com/voltvirtuoso/pylex#readme) and [changelog](https://github.com/voltvirtuoso/pylex/blob/main/CHANGELOG.md) for details.
```
