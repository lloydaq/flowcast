# FlowCast

**Predict traffic before it happens.** FlowCast forecasts highway traffic speed 15, 30, 45 and 60 minutes ahead using the METR-LA loop-detector dataset (207 sensors, Los Angeles, 5-minute resolution).

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Dataset acquisition, reconstruction, validation | ✅ done |
| 2 | Exploratory data analysis | next |
| 3–6 | Features, baselines, LightGBM, graph deep learning | planned |
| 7–10 | FastAPI backend, React dashboard, integration, release | planned |

## Setup (Windows / PowerShell)

Requires Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/prepare_metr_la.py   # downloads ~230 MB once, writes data/processed/
pytest                              # unit tests + data integration tests
ruff check .
```

The first run downloads the dataset from Hugging Face and caches it, which needs about 2.8 GB of disk. Later runs reuse the cache. Integration tests are skipped until `prepare_metr_la.py` has been run.

## Layout

```text
src/flowcast/data/metr_la.py      load windows, reconstruct canonical series, load sensor graph
src/flowcast/data/validation.py   data-quality checks and statistics
scripts/prepare_metr_la.py        end-to-end acquisition + validation pipeline
tests/                            unit tests (synthetic) and data integration tests
docs/dataset.md                   verified dataset documentation
```

## Data

See [docs/dataset.md](docs/dataset.md). Key facts: speeds are in **mph**, `0` means missing (8.1 % overall, mostly network-wide outages), and the published 70/10/20 chronological splits contain 23,974 / 3,425 / 6,850 samples.

## Citation

The dataset comes from Li, Yu, Shahabi & Liu, *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*, ICLR 2018. The Hugging Face packaging is by [witgaw](https://huggingface.co/datasets/witgaw/METR-LA).

## License

Code is released under the [MIT License](LICENSE). The METR-LA data is not redistributed here; it is downloaded from its source under its own (MIT) license.
