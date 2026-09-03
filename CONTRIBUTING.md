# Contributing to Pylex

Thank you for helping improve Pylex. Contributions that improve OCR reliability, searchable-PDF output, documentation, portability, testing, and reproducible benchmarks are welcome.

## Before opening an issue

Please search existing issues first. For an OCR problem, include the Pylex version, Python version, operating system, input format, image resolution, language, model size, inference engine, exact command or minimal Python code, and the observed versus expected result. Do not upload confidential documents; use a synthetic or redacted reproduction whenever possible.

## Development setup

```bash
git clone https://github.com/voltvirtuoso/pylex.git
cd pylex
python -m pip install -e ".[ocr,dev]"
pytest
ruff check .
```

## Pull requests

Keep pull requests focused and explain the user-visible behavior that changes. Add or update tests for behavior changes. Update the README, API documentation, CLI documentation, or changelog when the change affects users. Benchmark performance changes on a representative workload and document the hardware, model, runtime, and input characteristics.

By contributing, you agree that your contribution is provided under the repository’s MIT License.
