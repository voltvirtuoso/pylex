# Pylex Accuracy and Performance Benchmarks

Pylex performance and OCR quality depend on the document, resolution, language, model, runtime, operating system, and hardware. This page is a template for publishing reproducible measurements rather than a universal accuracy claim.

## Report every benchmark condition

Record the Pylex version, Python version, operating system, CPU/GPU, RAM, input format, page count, pixel dimensions or DPI, language, model size, inference engine, tiling options, worker count, warm-up state, and whether model compilation or cache creation is included.

## Suggested benchmark table

| Workload | Input | Model/runtime | Options | Cold time | Warm time | OCR quality metric |
|---|---|---|---|---:|---:|---:|
| Simple raster image | Describe image and resolution | PP-OCRv6 small / runtime | Default | TBD | TBD | TBD |
| Dense document | Describe document and resolution | PP-OCRv6 small / runtime | Tile size and overlap | TBD | TBD | TBD |
| Searchable PDF | Page count and source type | Model/runtime | DPI and workers | TBD | TBD | TBD |

## Measuring OCR quality

Use a representative, permission-cleared validation set and report the metric definition, such as character error rate or word error rate. Keep a separate set for evaluation so tuning does not overfit the published examples. For searchable-PDF workflows, test both text recognition and text-layer alignment on rotated pages.

Do not generalize from one document or one local machine. The purpose of this page is to help users choose a configuration and reproduce results on their own workload.
