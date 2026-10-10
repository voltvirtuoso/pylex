# Accuracy and benchmarks

Speed and quality depend on the document, resolution, language, model, runtime and hardware. Use this page to choose settings, and to publish measurements that others can reproduce.

## Choosing settings

| Goal | Start with | Trade-off |
|---|---|---|
| Best general accuracy | CLI default (`--model-size medium`) | More CPU time and memory |
| Balanced speed | `--model-size small` | Adaptive full-page pass for images up to 8 MP |
| Simple, clean image, fastest | `--fast` | Weaker on dense, rotated or noisy pages |
| Dense or large drawings | default tiling; try `--tile-size 1000` | More passes, better small-text recall |
| Text touching line art | `--det-box-thresh 0.45 --det-unclip-ratio 2.0` | More false positives |
| Faint or tiny text | `--retry-upscale`, higher `--dpi` | More time |
| Many files, CPU | `-j N` | Each worker owns an engine and memory |
| NVIDIA GPU | `--device gpu -j 1` | Needs a GPU profile; see Installation |

Note the default model differs: the CLI uses `medium`, while `Config()` in Python uses `small`.

## Measurements recorded by the project

These come from the author's notes (`runtime_research.md`, `CHANGELOG.md`), measured in a sandbox. They are directional, not guarantees.

| Workload | Runtime | Result |
|---|---|---|
| 1845x832 image, 102 lines, PP-OCRv6 small, orientation on | Paddle dynamic | 10.29 s |
| Same | ONNX Runtime | 0.81 to 0.95 s |
| Same | ONNX Runtime + OpenVINO | 1.06 s |
| A1 P&ID at 300 DPI, 2000 px tiles | OpenVINO | 24 tiles, 265 detections, 10.7 s |

OpenVINO was not faster than plain ONNX Runtime in these runs, so benchmark both on Intel hardware. **GPU timings have not been published**; measure on your own card.

## Comparison with other tools

`benchmarks/compare.py` builds scan-like test pages with known word positions, runs Pylex, OCRmyPDF with Tesseract, and OCRmyPDF with the `ocrmypdf-paddleocr` plugin from PyPI (`ocrmypdf --plugin ocrmypdf_paddleocr`; OCRmyPDF's own page also lists a different PaddleOCR plugin, so say which one you used), and scores each text layer:

- **Recall**: share of true words whose text appears in the layer, wherever it is.
- **Aligned**: share found in the right place (word centre within 0.6 line-heights vertically and 4 horizontally).
- **Extra**: OCR words that match no true word.

Cases: a plain text page, a dense page of small labels with line work, and the plain page stored sideways with `/Rotate 90` and `/Rotate 270`. Every test word carries a unique tag so chance matches stay near zero. The scoring itself is verified with a perfect-text-layer oracle at all four rotations (`--self-test`).

```bash
python benchmarks/compare.py                                   # every installed tool
python benchmarks/compare.py --only pylex --pylex-args "--device gpu"
python benchmarks/compare.py --input-dir my-scans/             # your own files: timing and word counts
```

### Result so far: OCRmyPDF with Tesseract

Measured on a single-core Linux sandbox, OCRmyPDF 17.13 with Tesseract 5 and the English model. Synthetic pages, so read these as a baseline for the harness, not as a verdict on any tool.

| Case | Seconds | Recall | Aligned | Extra |
|---|---:|---:|---:|---:|
| Plain text page | 4.8 | 99% | 99% | 1% |
| Dense labels and line work | 10.9 | 83% | 70% | 20% |
| Plain page, `/Rotate 90` | 4.9 | 100% | 100% | 0% |
| Plain page, `/Rotate 270` | 5.1 | 100% | 100% | 0% |

What this shows: in these tests OCRmyPDF places text correctly on rotated pages, so correct rotation handling looks like a baseline requirement, not a Pylex advantage. The place to look for a difference is dense, sparse-label pages, where Tesseract's page layout analysis struggled (70% aligned, 20% extra words).

**Pylex and the PaddleOCR plugin have not been run through it yet.** The environment this harness was developed in could not download PaddleOCR models, so those rows are empty rather than guessed. Run the command above on your machine and add the table here.

## Reporting your own benchmark

Record the Pylex, Python and OS versions; CPU or GPU and RAM; input format, page count and resolution; language, model size and runtime (`pylex doctor` prints most of this); tiling options, worker count; and whether the run was cold (model download, compilation) or warm.

| Workload | Input | Model / runtime | Options | Cold | Warm | Quality metric |
|---|---|---|---|---:|---:|---:|
| Simple image | | | default | | | |
| Dense document | | | tile size, overlap | | | |
| Searchable PDF | | | dpi, workers | | | |

## Measuring quality

Use a permission-cleared validation set and report a defined metric such as character or word error rate. Keep evaluation data separate from the data you tuned on. For searchable PDFs also test text-layer alignment on rotated pages. Do not generalize from one document or one machine.
