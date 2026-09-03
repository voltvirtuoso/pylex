# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-08-28

### Changed
- Renamed the distribution and command-line program from `pidocr` to `pylex`.
- Generalized the project from engineering drawings to broad OCR and document
  text extraction workloads.
- Added the reusable `pylex.api` library interface with `extract_text`,
  `create_engine`, and `create_searchable_document`.
- Exported `TextRegion`, `Config`, `FileResult`, and the document-processing
  functions from the `pylex` package root.
- Renamed internal environment variables and module imports to the Pylex
  namespace.

### Compatibility
- This is a major package rename. Update imports from `pidocr` to `pylex` and
  the executable from `pidocr` to `pylex`.

## [1.5.10] - 2026-08-27

### Improved
- Added targeted 90-degree retries for low-confidence tall OCR polygons,
  recovering vertical labels without rotating every page or doubling all OCR
  work.
- Added `--no-vertical-text-retry` as a diagnostic/performance override; the
  accuracy-first default keeps the retry enabled.

### Validation
- The uploaded screenshot's highlighted `01-ESDV-0101` and `S01-BDV-0107`
  labels were recovered by the targeted retry, whereas the normal pass alone
  misread or missed them.
- The attached Halini P&ID retained searchable-PDF output and was reprofiled
  with the vertical retry enabled.

## [1.5.9] - 2026-08-27

### Performance
- Increased the default tile size from 1600 to 2000 pixels for large P&ID
  pages. On the attached A1 Halini sample at 300 DPI, this reduced the tile
  count from 40 to 24 and produced 265 detections versus 276 with 1600-pixel
  tiles in the sandbox benchmark.
- Kept `--tile-size 1600` available for maximum dense-drawing local resolution
  and recall when the user prefers accuracy over throughput.

### Validation
- The attached `10020-01-PID-005.pdf` completed end to end with OpenVINO,
  generated a one-page searchable PDF, and added 265 OCR lines in 10.7 seconds
  in the sandbox. Windows timing depends on CPU, memory, and antivirus behavior.

## [1.5.8] - 2026-08-27

### Improved
- Added a persistent OpenVINO compiled-engine cache with a platform-specific
  default and the `--runtime-cache-dir` override.
- Verbose startup now identifies the selected runtime so first-run compilation
  can be distinguished from recurring OCR work.

### Note
- The first process can remain slower while models and kernels are prepared;
  the cache is intended to reduce repeated engine-construction overhead in
  later processes, not eliminate model loading or Windows filesystem overhead.

## [1.5.7] - 2026-08-27

### Changed
- Added explicit `openvino` runtime routing through PaddleX's supported
  ONNX Runtime runner with `OpenVINOExecutionProvider`.
- Automatic runtime selection now prefers OpenVINO when its provider is
  available, then standard ONNX Runtime, then dynamic Paddle.
- Separated the `onnx` and `openvino` optional dependencies so Windows users
  do not install conflicting ONNX Runtime distributions.

### Performance
- On an equivalent 1845×832 image with 102 OCR lines, OpenVINO completed
  prediction in about 1.06 seconds and standard ONNX Runtime in about 0.81–0.95
  seconds, versus about 10.29 seconds for dynamic Paddle in the sandbox.
  Target-CPU benchmarking remains recommended because OpenVINO is optimized
  for Intel hardware and is not universally faster than standard ONNX Runtime.

## [1.5.6] - 2026-08-27

### Fixed
- Progress checkpoints are now monotonic: completing a tile advances the bar
  to its completed-work position, and PDF assembly cannot reset it backward.
- A live spinner now runs during Paddle's monolithic `predict()` call, where
  Paddle exposes no intra-inference progress callback. The percentage remains
  truthful instead of pretending to measure internal neural-network layers.
- Standalone-image and PDF callbacks now share the same page/tile progress path.

## [1.5.4] - 2026-08-27

### Fixed
- Progress now advances fractionally during page rendering and OCR tile
  inference, so a one-file run no longer stays at 0% until jumping to 100%.
- PDF page stages now use the same live progress path as standalone images.

## [1.5.3] - 2026-08-27

### Fixed
- Added adaptive full-page inference for PP-OCRv6 small images that fit within
  the detector resolution and pixel budget, avoiding unnecessary overlapping
  tile passes without changing the OCR model or thresholds.
- Added explicit single-image OCR stage and tile-progress status so long image
  inference is visible instead of appearing stuck at 0%.

## [1.5.2] - 2026-08-27

### Fixed
- Added explicit engine-ready timing and model/runtime information to the
  sequential progress display, so model initialization no longer looks frozen.
- Added post-engine image-stage and per-tile progress callbacks for single-file
  OCR, including visible `OCR tile N/M` status before each blocking inference.

## [1.5.1] - 2026-08-27

### Fixed
- Removed `cpu_threads` from Paddle dynamic runner configuration; dynamic
  Paddle rejects that field, so thread limits are now passed only to static
  and ONNX Runtime engines while native thread-pool limits remain available.
- Changed optimized-runtime constructor fallback to rebuild directly with
  `paddle_dynamic` instead of retrying the incompatible static oneDNN path.
- Added runtime regression coverage and an end-to-end medium-model CLI smoke
  test.

## [1.5.0] - 2026-08-27

### Changed
- Made PaddleOCR the only supported OCR backend; removed the optional Windows
  OCR adapter and dependency path.
- Switched the default model generation from PP-OCRv4 to PP-OCRv6 small and
  made Paddle dynamic CPU inference the verified cross-platform default.
- Added explicit PP-OCR model-tier selection (`tiny`, `small`, `medium`) and
  Paddle runtime selection (`paddle_static`, `paddle_dynamic`, `onnxruntime`).
- Added PaddleOCR high-performance inference controls, including optional HPI
  backend selection and a documented runtime fallback for unsupported runtimes.
- Added recognition and text-line orientation batch-size controls.

### Improved
- Preserved the tiled accuracy-first workflow while removing the requirement
  to use `--fast` just to avoid an unnecessarily old or slow default stack.
- Added explicit medium-tier selection for users who want the highest Paddle
  model accuracy, while keeping the default tuned for practical CPU latency.
- Updated the OCR dependency range to PaddleOCR/PaddlePaddle 3.x to match the
  current constructor and inference-engine API.

## [1.4.0] - 2026-08-27

### Added
- Added `--fast` for simple images: one full-image pass, no text-line
  orientation classifier, and a 3200 px detector cap.
- Added `--retry-upscale` for a single 1.5x recovery pass when no usable text
  is found.
- Added automatic CPU thread budgeting across `-j` worker processes and an
  explicit `--cpu-threads` override.

### Improved
- Avoided implicit negative-stride image copies before Paddle inference.
- Kept the accuracy-first tiled workflow as the default for large/dense P&ID
  drawings while making the latency/recall trade-off explicit for simple images.

## [1.3.2] - 2026-08-27

### Fixed
- Suppressed the misleading Windows `INFO: Could not find files for the
  given pattern(s).` message emitted when Paddle checks whether optional
  `ccache` tooling is installed. This message is from Paddle’s startup probe,
  not from pylex’s `-xf` matcher.

## [1.3.1] - 2026-08-27

### Fixed
- Multi-worker and sequential runs now show startup activity while PaddleOCR
  imports, initializes, and performs its first model/backend setup, instead of
  leaving the progress display at 0% until the first file completes.
- PaddleOCR/PaddleX status output is suppressed around engine construction and
  prediction, preventing backend chatter from drowning out pylex progress
  output while preserving pylex warnings and errors. Windows OS-level output
  redirection also handles status emitted by backend child processes.
- Clarified `-xf` wildcard usage for suffix-style exclusions: use patterns
  such as `*_OCR*` to skip previously generated OCR outputs and prevent names
  such as `drawing_OCR_OCR.pdf`.


## [1.3.0] - 2026-08-27

### Added
- `-xf`/`--exclude`: skip files matching a glob/wildcard pattern when
  scanning folders (fnmatch rules — `*` matches anything, including path
  separators, so both `*_draft*` and `backup/*` work as expected).
  Matched against the filename and the path relative to the input folder.
  Repeatable, and comma-separable within one flag.
- Live progress bar (`pylex.progress`, no external dependency) tracking
  files processed, with the current file — and page, for a large
  multi-page PDF — shown as a trailing suffix. Falls back to plain
  per-file lines when stdout isn't a real terminal (piped/redirected
  output), so a log file doesn't fill with carriage-return noise.
- `--dry-run` now also reports which `-xf` patterns were applied.

### Fixed
- `-q`/`--quiet` now actually suppresses per-file status output (it only
  filtered log messages before); failures still surface as `log.error`
  either way.

## [1.2.0] - 2026-08-27

### Added
- Tiled OCR (`pylex.tiling`): pages are OCR'd in overlapping windows at
  full local resolution by default (`--tile-size 1600 --tile-overlap 220`),
  then merged and de-duplicated back into page coordinates. Fixes a
  whole-page pass missing tags squeezed into dense symbol clusters (a
  valve/instrument/line-tag crowd) even though the same text on open pipe
  run OCRs fine — the whole-page resize before detection was shrinking
  small dense-area text below the detector's threshold.
- `--tile-size`, `--tile-overlap`, `--dedupe-tol-px`, `--no-tile` flags to
  control/disable tiling.
- Detector-tuning passthrough flags: `--det-thresh`, `--det-box-thresh`,
  `--det-unclip-ratio` (map to PaddleOCR's `text_det_*` knobs; only sent
  when set, with a graceful fallback if an installed PaddleOCR version
  doesn't accept them).
- `CHANGELOG.md` (this file).

### Changed
- `pyproject.toml` now sources `version` dynamically from
  `pylex.__version__` (`[tool.hatch.version]`) instead of duplicating the
  version string in two places.

## [1.1.0] - 2026-08-27

### Added
- `-j`/`--jobs N`: process N files in parallel using separate worker
  processes (`pylex.runner`), each loading its own OCR engine once and
  reusing it across every file that worker picks up. Parallelism is
  across files, not pages within one file.
- Warning when `--gpu` is combined with `-j > 1` (workers would contend
  for the same GPU).
- Info note (instead of a silent cap) when `-j` exceeds the detected CPU
  core count.

## [1.0.0] - 2026-08-25

### Added
- Initial `pylex` CLI package, restructured from a working single-file
  PaddleOCR proof of concept into an installable `src/`-layout package
  (`pyproject.toml`, console-script entry point, `python -m pylex`).
- Core OCR pipeline: render a PDF page as the viewer displays it, run
  PaddleOCR, transfer the page's `/Rotate` into its content stream, and
  merge an invisible text layer anchored at the start of each detected
  polygon. This combination is what makes OCR placement correct on
  rotated pages (`/Rotate` 90/180/270) as well as upright ones.
- Support for PDF and image (`.png`, `.jpg`/`.jpeg`, `.tif`/`.tiff`,
  `.bmp`, `.webp`) input, single files, multiple files, or folders
  (`-r` to recurse).
- `-o/--output`, `--in-place` (gated behind `--force`), `--suffix`,
  `--force`, `--skip-existing`, `--formats`, `--pages` (tesseract-style
  page ranges), `--dry-run`.
- OCR/overlay tuning flags: `--dpi`, `--image-dpi`, `--lang`,
  `--min-confidence`, `--det-limit-side-len`, `--gpu`,
  `--no-textline-orientation`, `--font`, `--baseline-fraction`.
- Unit tests for page-range parsing, quad geometry, and output-path
  resolution.

[Unreleased]: https://github.com/voltvirtuoso/pylex/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/voltvirtuoso/pylex/compare/v1.5.10...v2.0.0
[1.5.10]: https://github.com/voltvirtuoso/pylex/compare/v1.5.9...v1.5.10
[1.5.9]: https://github.com/voltvirtuoso/pylex/compare/v1.5.8...v1.5.9
[1.5.8]: https://github.com/voltvirtuoso/pylex/compare/v1.5.7...v1.5.8
[1.5.7]: https://github.com/voltvirtuoso/pylex/compare/v1.5.6...v1.5.7
[1.5.6]: https://github.com/voltvirtuoso/pylex/compare/v1.5.5...v1.5.6
[1.5.5]: https://github.com/voltvirtuoso/pylex/compare/v1.5.4...v1.5.5
[1.5.4]: https://github.com/voltvirtuoso/pylex/compare/v1.5.3...v1.5.4
[1.5.3]: https://github.com/voltvirtuoso/pylex/compare/v1.5.2...v1.5.3
[1.5.2]: https://github.com/voltvirtuoso/pylex/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/voltvirtuoso/pylex/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/voltvirtuoso/pylex/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/voltvirtuoso/pylex/compare/v1.3.2...v1.4.0
[1.3.2]: https://github.com/voltvirtuoso/pylex/compare/v1.3.1...v1.3.2
[1.3.1]: https://github.com/voltvirtuoso/pylex/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/voltvirtuoso/pylex/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/voltvirtuoso/pylex/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/voltvirtuoso/pylex/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/voltvirtuoso/pylex/releases/tag/v1.0.0
