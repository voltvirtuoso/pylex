# Pylex Python API: OCR Regions and Searchable PDFs

Pylex can be used as a Python library as well as a command-line program. The public API exposes OCR regions with recognized text, confidence, and polygons in the original image pixel coordinates.

## Extract OCR regions

```python
from pylex import Config, create_engine, extract_text

config = Config(inference_engine="auto", model_size="small")
engine = create_engine(config)
regions = extract_text("page.png", config=config, engine=engine)

for region in regions:
    print({
        "text": region.text,
        "confidence": region.confidence,
        "polygon": region.polygon.tolist(),
    })
```

Reuse the same engine for multiple files to avoid rebuilding the OCR backend for each call.

## Create a searchable PDF

```python
from pylex import Config, create_engine, create_searchable_document

config = Config(inference_engine="auto")
engine = create_engine(config)
result = create_searchable_document(
    "input.pdf",
    "output-searchable.pdf",
    config=config,
    engine=engine,
)

if not result.ok:
    raise RuntimeError("Pylex could not create the searchable PDF")

print(result.pages_processed, result.lines_added)
```

## Progress callbacks

Applications can provide `on_page`, `on_stage`, `on_tile`, and `on_tile_done` callbacks to integrate processing progress into a service, notebook, or custom interface. The lower-level `process_image` and `process_pdf` functions are available when direct output-path or callback control is required.

## API expectations

Pylex returns OCR detections rather than a guaranteed semantic document model. Confidence is an OCR signal, not a proof of correctness. Use a representative validation set when integrating Pylex into production workflows.
