# Tiered federated learning for ECG classification

This repository contains reproducible research code for ECG classification across
clients with different sensing capabilities. It studies how to combine information
from clinical ECG sources with models intended for wearable and single-lead devices.

The code implements a round-based federated comparison using validation-gated
aggregation and a personalised readout for each device tier, compared with a FedAvg
baseline.

The experiment uses an echo state network (ESN) as a fixed feature extractor and fits a
regularised linear readout. The readout is trained with a closed-form solve; the project
does not require a deep-learning framework. This is research software and the IoT ECG
signals are simulated from clinical recordings, not collected from wearable devices.

## Dataset

The experiments use the [PhysioNet/Computing in Cardiology Challenge 2021 training
set](https://physionet.org/content/challenge-2021/1.0.3/). It contains multiple ECG
corpora with different sources and populations. The current configurations use these
corpus directories:

| Directory | Use in the configurations |
|---|---|
| `ptb-xl` | Hospital tier |
| `georgia` | Hospital tier |
| `cpsc_2018_extra` | Hospital tier |
| `ningbo` | Wearable tier |
| `chapman_shaoxing` | Wearable tier |
| `st_petersburg_incart` | Wearable tier |
| `cpsc_2018` | IoT tier, degraded to a single-lead signal |

The dataset is not included in this repository. Download it from PhysioNet and follow
the dataset's access and use terms. The full training set is approximately 17 GB.
From the repository root, it can be downloaded with:

```bash
wget -r -np -P data https://physionet.org/files/challenge-2021/1.0.3/training/
```

The loader accepts either the download root or the `training/` directory as the data
root. Records are read from the Challenge's header and signal files. You can set another
location in a YAML configuration or override it with `--data-root`.

## Code and methodology

### Round-based experiment

The primary pipeline is under `src/tieredfl/` and is configured by YAML files in
`configs/`. The default full configuration defines nine clients across the hospital,
wearable, and IoT tiers. Clients draw from different data sources, creating a naturally
non-IID setting. Records are allocated disjointly between clients by default; this is
record-level partitioning, not patient-level partitioning.

Each record is resampled to 500 Hz, adjusted to a fixed length, baseline-corrected,
band-limited, and standardised per lead. The IoT tier is simulated by keeping one lead,
downsampling it to 250 Hz, and adding configurable Gaussian noise. A fixed sparse ESN
converts each signal into pooled features. Clients fit ridge-regression readouts locally.

In each communication round, the server forms two candidate readouts: sample-weighted
FedAvg and per-class selective aggregation. A validation gate chooses an interpolation
between them to maximise the configured objective while keeping shared-model regression
within a fixed tolerance when a candidate meets that constraint. If none meets it, the
gate selects the best-scoring candidate and reports that the constraint was unmet. The
personalised tier readout is fitted locally on each tier's clients; only readout weights
are averaged for that head. The standard run reports these methods, reference baselines,
and per-tier metrics.

Important modules:

| Path | Purpose |
|---|---|
| `src/tieredfl/data/` | Record loading, preprocessing, labels, and IoT degradation |
| `src/tieredfl/model/` | Reservoir feature extractor and ridge readout |
| `src/tieredfl/federated/` | Client updates, aggregation, validation gate, and tier heads |
| `src/tieredfl/evaluation/` | Metrics and reference baselines |
| `configs/` | Smoke, full-scale, and legacy run configurations |

## Requirements

- Python 3.9 or later
- NumPy 1.24 or later
- SciPy 1.10 or later
- PyYAML 6.0 or later
- `wget` to download the dataset using the command above

The optional development dependencies are pytest and Ruff. Install them with the
`dev` extra shown below.

## Usage

Run the commands below from the repository root.

### 1. Install the project

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

### 2. Download and inspect the data

Download the Challenge data as described in the [Dataset](#dataset) section. Then check
whether the configured record counts are available:

```bash
.venv/bin/tieredfl inspect --config configs/smoke.yaml --data-root data
```

The default `data_root` is `data`. Use `--data-root` if the files are elsewhere.

### 3. Run the round-based experiment

The smoke configuration exercises the pipeline on a small subset. Its scores are for
pipeline checks and should not be treated as research results.

```bash
./run.sh
```

Run the full configuration, or override the number of rounds:

```bash
./run.sh full
./run.sh full --rounds 10
```

The script writes a timestamped log under `logs/` and a machine-readable JSON result
under `results/`. To use the CLI directly:

```bash
.venv/bin/tieredfl run --config configs/full.yaml
.venv/bin/tieredfl cache --config configs/full.yaml
```

`cache` extracts and stores fixed-reservoir features, then exits. The cache is keyed by
the settings that affect feature extraction and can be reused when changing rounds or
the aggregation gate. Useful run options include `--scale 0.1`, `--rounds 10`,
`--seed 42`, `--personal-weight 0`, and `--noise-sigma 0.01`. Use
`.venv/bin/tieredfl run --help` for the full option list.

## Results and reproducibility

Each run is determined by its YAML configuration and seed. The round-based run saves
the effective configuration with its result. Record selection, IoT noise, and optional
transmission noise are seeded. Feature extraction is cached under `.feature_cache/` by
default.

Seeded runs are intended to be repeatable under a fixed software and numerical
environment. Exact results can differ across implementations or numerical backends.
This repository is a port of an earlier Kotlin/JVM implementation; its figures are not
bit-identical to the earlier runs because the random streams and linear-algebra backend
differ. Recompute and cite figures from the implementation that produced them.

Two interpretation points matter when reporting round-based results:

- **Present-class macro F1** averages over classes that occur in the evaluated split.
  Plain macro F1 averages over all five configured classes, including absent ones.
- The default `personal_weight` is `1.0`, so IoT evaluation uses the tier head. At this
  setting, the IoT scores do not measure the validation gate. Use `--personal-weight 0`
  to evaluate the gate under the shared model, and compare the tier head against the
  supplied IoT baselines to assess personalisation. See [`docs/notes.md`](docs/notes.md)
  before interpreting default-run results.

The IoT baselines use the same evaluation split:

| Baseline | What it measures |
|---|---|
| `majority` | Always predicting the most frequent class present in the split |
| `restricted_global` | The shared model with predictions restricted to present classes |
| `local_only_mean` | Mean performance of client heads before they are aggregated |
| `federated` | Performance of the aggregated IoT tier head |

Compare IoT gains with `majority` and `restricted_global`, and use `local_only_mean` to
check whether aggregation improves on leaving each client head local.

## Citations

If you use the Challenge data, cite its versioned PhysioNet record:

> Reyna, M., Sadr, N., Gu, A., Perez Alday, E. A., Liu, C., Seyedi, S., Shah, A., &
> Clifford, G. D. (2022). *Will Two Do? Varying Dimensions in Electrocardiography: The
> PhysioNet/Computing in Cardiology Challenge 2021* (version 1.0.3). PhysioNet.
> https://doi.org/10.13026/34va-7q14

For the software citation, use [`CITATION.cff`](CITATION.cff).

## License and contributions

The code is released under the MIT License; see [`LICENSE`](LICENSE). The dataset has
separate access and use terms, which remain in effect.

Contributions are welcome through issues or pull requests. For changes that affect
results, include the configuration and seed used, explain the methodological impact, and
add or update focused tests where appropriate.
