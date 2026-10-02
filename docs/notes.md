# Implementation notes

Decisions, departures from the earlier Kotlin implementation, and known limitations.
Anything here that affects a number is called out as such.

## Departures that change results

### Clients sharing a dataset get disjoint records

`cpsc_2018` holds 6877 usable records. The original configuration gave its three IoT
clients 3000, 2500 and 2000 records, sampling each client independently from the shared
pool — 7500 draws from 6877 records. Clients therefore overlapped, and a recording could
appear in one client's training split and another's validation split. Since the global
validation set is the union of all clients' validation splits, some training data was
also being evaluated against.

`configs/full.yaml` allocates disjoint records and caps the IoT clients at 2800, 2300
and 1700. `configs/legacy-overlap.yaml` sets `disjoint_clients: false` and restores the
original counts, for reproducing the earlier results.

This affects reported figures. Expect the IoT tier to look worse under disjoint
allocation, since it is no longer partly evaluated on its own training data.

### Record selection and injected noise are seeded

The original used `dataset.shuffled()` and `Random.Default`, neither seeded, so no run
could be reproduced exactly. Everything now derives from the run seed.

Seeds are derived through `seeding.stable_hash`, which uses blake2b rather than Python's
`hash()`. String hashing in CPython is salted per process, so a seed derived from it
changes between interpreter invocations.

## The gate's objective does not reach the number it is named after

This is worth understanding before reporting anything from a default run.

The gate maximises `iot_present_macro_f1` **of the shared global model**. The IoT tier is
then evaluated under `evaluation_head(global_wout, tier_head, personal_weight)`. At the
default `personal_weight: 1.0` that function returns the tier head and ignores the global
model entirely. The tier head is fitted once, before either arm starts training, and does
not change between rounds.

So at `personal_weight: 1.0`:

- every reported IoT figure is a property of the tier head alone;
- IoT figures are **identical** in the gated and FedAvg arms, and **constant** across
  rounds;
- the gate's choice affects only the global, hospital and wearable rows.

A five-round smoke run makes this plain — IoT present macro F1 is `0.6410` in all ten
round records across both arms, while global macro F1 varies from 0.230 to 0.271:

```
gated   iot present_macro_f1: [0.6410, 0.6410, 0.6410, 0.6410, 0.6410]
        global macro_f1:      [0.2512, 0.2375, 0.2709, 0.2320, 0.2395]
fedavg  iot present_macro_f1: [0.6410, 0.6410, 0.6410, 0.6410, 0.6410]
        global macro_f1:      [0.2653, 0.2375, 0.2690, 0.2300, 0.2445]
```

The consequence: any IoT improvement reported from a default run comes from the tier
head, not from the gate. The two mechanisms cannot both be credited for it, and the gate
cannot be evaluated on IoT numbers at this setting at all.

This behaviour is inherited from the Kotlin implementation, where `personalBeta`
defaulted to 1.0 and `cleanIotHead` was likewise computed once before both experiment
arms. It is faithful, not a port artefact.

Ways to get an honest reading of each mechanism:

- **To evaluate the gate**, set `personal_weight: 0.0`. The IoT tier is then evaluated
  under the shared model, and the gate's objective reaches the reported number.
- **To evaluate the tier head**, compare `federated` against `restricted_global` and
  `local_only_mean` in the ablation block, which is what those baselines are for.
- **To report both**, run the `personal_weight` sweep and show the curve; the endpoints
  separate the two contributions.

## What a first run shows

One run of `configs/full.yaml` at `--scale 0.08`, 30 rounds, seed 0, 34 minutes. **One
seed at 8% scale** — indicative only, and not a basis for any claim. Two results are
pointed enough to be worth chasing at full scale.

**The gated arm did not beat FedAvg on anything it can affect.**

```
global macro_f1      0.4186 vs 0.4329   -0.0143
wearable macro_f1    0.4611 vs 0.5139   -0.0528
hospital macro_f1    0.3901 vs 0.3800   +0.0101
iot present_macro_f1 0.8610 vs 0.8610    0.0000
```

The chosen weight averaged 0.56 and used the full range, so the gate was active, not
degenerate. It traded global and wearable performance for a small hospital gain, and the
IoT row it was optimising for is pinned by the tier head. On this run the gate is a net
loss. Worth testing at `personal_weight: 0.0`, where its objective reaches the IoT
numbers and the trade might pay.

**Federating the tier heads added little over local training.**

```
shared_global          present_macro_f1 0.4500
restricted_global      present_macro_f1 0.5670
local_only[iot_cpsc_a] present_macro_f1 0.8042
local_only[iot_cpsc_b] present_macro_f1 0.8054
local_only[iot_cpsc_c] present_macro_f1 0.9166
local_only_mean        present_macro_f1 0.8421
federated              present_macro_f1 0.8610
```

The federated head beats the mean of the local-only heads by 0.019 — and one client's
own head, trained alone on 107 samples, beats the federated head outright. This is the
comparison `local_only_mean` exists to force: on this run, personalising the tier is
what helps, and federating the personalised heads is close to free of benefit.

Note also `iot macro_f1 = 0.3444` against `iot present_macro_f1 = 0.8610`. Only two of
five classes appear in the IoT split, so the five-class average is roughly the
present-class figure times two fifths. Quoting either number without saying which is how
this tier gets misreported.

## Observations worth following up

### The ridge penalty is large relative to the pooled features

Pooled states are time averages of reservoir states. Driven by a z-scored,
approximately zero-mean signal, the reservoir's states oscillate around zero, so their
time average is small — on the order of 0.03 in magnitude for the configurations here.
The Gram matrix diagonal then averages around 0.03, which is the same order as the
default `ridge = 1e-2`.

The readout is therefore heavily shrunk. On a synthetic two-class task built from these
features, accuracy drops from 0.83 at `ridge = 1e-6` to 0.54 at `ridge = 1e-2`. This
behaviour is inherited from the original implementation, which used the same pooling and
the same penalty, and it is not a port artefact — but it suggests the penalty is worth
sweeping, or the features worth rescaling before the solve.

A related point: mean pooling discards phase and much of the waveform shape. It retains
the DC level and low-order nonlinear mixtures. Concatenating other statistics of the
state trajectory, or pooling over windows rather than the whole record, would give the
readout more to work with.

### The gate does move, and the trade it makes is small

On a run at 8% scale the chosen weight settles at 0.9–1.0 after the first couple of
rounds, with most of the grid feasible. The trade is favourable on the gate's own terms:
global macro F1 falls by about 0.001 while the global model's IoT present macro F1 rises
by about 0.10.

```
round 3 gate: weight=1.0 (11/11 feasible) global_macro_f1=0.3298->0.3287 \
              iot_present_macro_f1=0.3000->0.4030
```

So the mechanism is active rather than degenerate. Whether it matters to the
*reported* IoT numbers is a separate question — see the section above.

### Ties in the gate resolve to the lowest weight

When several mixtures score identically on the objective and all tie-breakers, the first
is taken, which is the one closest to FedAvg. This is intentional — prefer the less
aggressive aggregate when nothing distinguishes them — but it means a reported weight of
0.6 can mean "0.6 through 1.0 were equally good", not "0.6 was uniquely best".

## Known limitations

### Partitioning is record-level, not patient-level

Records are allocated disjointly, but a patient contributing several recordings can have
them split across clients or across a train/validation boundary. The earlier codebase
had a patient-id resolver for its hierarchical experiments; it is not used here, and
patient-level disjointness is not enforced.

For PTB-XL this matters: the source contains repeat recordings of the same subjects.
Figures should be read as record-level, and a patient-disjoint variant is the obvious
next step.

### The IoT tier is simulated, not measured

No wearable data is involved. The IoT tier is produced by taking lead I of a clinical
recording, halving the sampling rate, and adding Gaussian noise scaled to a fraction of
the signal's standard deviation. This captures lead reduction and rate reduction, and
approximates amplitude noise. It does not model electrode contact loss, baseline drift
from motion, saturation, or the burst-like structure of real motion artefact.

The degradation parameters are the sensitivity knobs: `iot.noise_amplitude` and
`iot.target_rate`.

### Transmission noise is not a privacy guarantee

`transmission_noise_sigma` adds isotropic Gaussian noise to transmitted readouts. It
probes how much perturbation the protocol tolerates. It is not a differential privacy
mechanism: no sensitivity bound is computed, no privacy budget is tracked, and no
epsilon can be claimed from it.

### The gate reads validation metrics centrally

The gate is evaluated against client validation states pooled on the server. In a
deployment the clients would compute and report these per-class scalars instead. The
quantities the gate consumes are scalars either way, so the mechanism transfers, but the
implementation as written does hold validation states centrally. The per-tier
personalised head does not — it receives only weight matrices and sample counts.

### Single seed per run

Each run uses one seed. Comparing the gated arm against the baseline across a single
seed gives no error bars. Repeat across seeds and report a spread before treating a
delta of a few thousandths as real.

## Environment

numpy built against Apple Accelerate raises spurious floating point warnings from inside
BLAS calls — a bare `a.T @ a` on random data triggers one. They carry no information
about the arrays involved and are filtered in the pytest configuration. Installing numpy
against OpenBLAS avoids them.
