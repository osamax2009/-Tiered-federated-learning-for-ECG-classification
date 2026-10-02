# Tiered federated learning for ECG classification

Federated learning across device tiers that differ in what they can physically record.
A hospital ECG captures twelve leads at 500 Hz; a wrist wearable captures one noisy lead
at 250 Hz. Training one shared model over both leaves the constrained tier evaluated
under a readout shaped largely by signals it could never acquire — and the constrained
tier is usually the deployment target.

This repository implements and evaluates two mechanisms for that problem:

- **Validation-gated aggregation.** Each round the server interpolates between weighted
  FedAvg and per-class selective aggregation, and keeps the mixture that maximises a
  stated objective for the weak tier, subject to the shared model not regressing beyond
  a fixed tolerance.
- **A per-tier personalised readout.** Clients in a tier fit a readout locally, transmit
  only the weights, and the server averages them. Patient-derived representations never
  leave the device.

Classification uses an echo state network: a fixed sparse reservoir as the feature map
and a ridge-fitted linear readout. Only the readout is trained, so a client's local step
is one linear solve and the per-round payload is a 5×1001 matrix — about 40 KB.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Requires Python 3.9+, numpy and scipy. No deep learning framework.

## Data

Uses the [PhysioNet/CinC Challenge 2021](https://physionet.org/content/challenge-2021/)
training set, which is not redistributed here. The full set is about 17 GB.

```bash
wget -r -np https://physionet.org/files/challenge-2021/1.0.3/training/
```

Point `data_root` at either the download root or the `training/` directory itself. Check
what was found before committing to a long run:

```bash
.venv/bin/tieredfl inspect --config configs/full.yaml --data-root /path/to/data
```

Seven sources are used, one per client, which is where the non-IID structure comes from:
they differ in recruitment, demographics and label prevalence, so the partition is
natural rather than synthetic.

## Run

```bash
./run.sh                    # smoke config, a few minutes, checks the pipeline works
./run.sh full               # full scale, 30 rounds
./run.sh full --rounds 10   # trailing arguments pass through to the CLI
```

`run.sh` logs to `logs/<config>_<timestamp>.log` and the run writes a JSON artefact to
`results/`. Build tables and figures from the JSON rather than by parsing logs.

Or drive the CLI directly:

```bash
.venv/bin/tieredfl run --config configs/full.yaml
.venv/bin/tieredfl cache --config configs/full.yaml      # extract features and stop
```

Useful overrides: `--scale 0.1` (fraction of each client's records), `--rounds`,
`--seed`, `--personal-weight` (0 = shared model, 1 = pure tier head), `--noise-sigma`.

### Feature caching

The reservoir is fixed, so a client's pooled states never change between rounds.
Extraction therefore runs once per configuration and is cached under `.feature_cache/`,
keyed by a hash of everything that affects it. This dominates total cost: the first run
takes hours at full scale, and subsequent runs over the same features take minutes. The
cache key deliberately excludes rounds, the gate objective and the personalisation
weight, so sweeping those reuses the same features.

## Layout

```
src/tieredfl/
  data/         record reading, clinical preprocessing, IoT degradation, partitioning
  model/        reservoir and ridge readout
  federated/    clients, aggregation rules, the gate, per-tier personalisation
  evaluation/   metrics and the IoT reference baselines
  experiment.py the round loop
  report.py     JSON artefacts and the console summary
configs/        run definitions
tests/          unit tests
```

## Reading the results

Two macro averages are reported. Plain macro F1 averages over all five classes, which
penalises a tier for classes absent from its own validation split. **Present-class macro
F1** averages only over classes that actually occur in the split being scored, and is
the figure to compare across tiers.

Every run scores four reference points for the IoT tier on the identical split, because
a five-class accuracy figure for that tier is easy to misread:

| baseline | what it establishes |
|---|---|
| `majority` | the floor: always predict the most frequent present class |
| `restricted_global` | the shared model with argmax restricted to present classes |
| `local_only_mean` | each client's own head, unaggregated — if this matches `federated`, federating bought nothing |
| `federated` | the aggregated tier head |

Report gains against `majority` and `restricted_global`, not against an unrestricted
five-way argmax.

> **Before reporting a default run**, read
> [the first section of docs/notes.md](docs/notes.md). At `personal_weight: 1.0` the IoT
> tier is evaluated under the tier head, which is fitted once before training — so IoT
> figures are identical across the gated and baseline arms and constant across rounds,
> and the gate's objective does not reach them. Use `--personal-weight 0` to evaluate
> the gate, and the ablation block to evaluate the tier head.

## Reproducibility

A run is determined by its config file and seed. Record selection, IoT noise and
transmission noise all derive from the run seed through `seeding.py`, which uses a
content hash rather than `hash()` — the latter is salted per process and would make runs
irreproducible.

The reservoir's spectral-radius rescaling needs care: ARPACK draws a random start vector
unless given one, and the resulting estimate varies between calls by enough to change
the rescaled weights in the sixth decimal, which is visible in the pooled states a few
thousand steps later. The start vector is seeded for this reason.

## Relationship to the earlier Kotlin implementation

This is a port of a Kotlin/JVM implementation, reorganised around one configurable
experiment rather than a file per variant. The algorithms, equations and hyperparameters
match; the numbers will not be bit-identical, because the random streams differ and the
linear algebra now runs through BLAS. Published figures from the earlier runs should be
cited to that implementation, not to this one.

Three deliberate departures, all documented in [docs/notes.md](docs/notes.md):

1. **Clients sharing a dataset get disjoint records.** The original sampled each client
   independently from the shared pool, so the three IoT clients overlapped — a recording
   could appear in one client's training split and another's validation split.
   `configs/legacy-overlap.yaml` reproduces the original behaviour.
2. **Record selection and IoT noise are seeded.** They were not before.
3. **Features are cached to disk** rather than recomputed every run.

## License

MIT — see [LICENSE](LICENSE).
