# Pango FPGA SRB Delay Estimator

C++17 solution for the 2026 EDA Elite Challenge problem "Routing Delay Estimation for
Ultra-Large-Scale FPGAs" from Pango.

The submitted path is a deterministic gradient-boosted estimator trained from the public
one-million-query answer set. A separate exact implicit-graph router remains in the project as
an oracle for architecture validation and golden regression tests.

## Current Algorithm Status

The repository contains two intentionally separate runtime policies:

- **Latest accuracy candidate:** `--short-route`. It uses the architecture-driven weighted local
  router for Manhattan distance `0--32` and achieved `95.0229` on the locked public final block.
- **Conservative baseline:** no extra flag. It keeps the original 128-tree estimator and scored
  `94.2958` on the same block. It remains available as a control while the full 100-million-row
  target-machine comparison is not yet complete.

Both modes are implemented in the tracked C++ source and covered by the CLI tests. The score above
is a public qualification result, not a guarantee of the hidden evaluation score.

## Build

Requirements:

- CMake 3.16 or newer
- A C++17 compiler (tested with GCC 13.3)
- Ninja or Make

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Build from the source tree before running either policy. The `build/` directories already present
on a developer machine are local, ignored artifacts and are not part of the repository.

## Contest Interface

Run from the project directory, where `arch/` contains the five architecture JSON files.

The conservative baseline uses the official no-flag interface:

```bash
build/estimate -in delay_estimate_request.csv -out delay_estimate_result.csv
```

An explicit architecture directory is also supported:

```bash
build/estimate -in request.csv -out result.csv --arch /path/to/arch
```

Without `--arch`, the executable first checks `./arch` and then an `arch/` directory beside the
executable. A relocatable package can be created with:

```bash
cmake --install build --prefix submission
```

The request is read and the result is written line by line. The program does not load the full
query set into memory. Input and output paths must differ.

To run the current strongest accuracy candidate, add `--short-route`:

```bash
build/estimate -in request.csv -out result.csv --short-route
```

It scores `96.8621` on distance `0--16` and `97.0510` on `17--32` in the locked final block, while
the complete final score improves from `94.2958` to `95.0229`. Five local one-million-row runs
average `16.16 s` and are byte-identical. Queries above 32 keep the original estimator prediction.

The earlier bounded learned residual remains available as `--short-residual`, with optional Pareto
thresholds selected by `--short-residual-threshold 32|64|96|128`. The no-flag production output is
unchanged. See `docs/RESULTS.md` for protocols, segment metrics, runtime, and determinism details.

## Validation

Validate architecture files and fixed invariants:

```bash
build/estimate --validate
```

Verify every Arc/Net in the official golden Paths, then independently prove their minimum delays
with exact A*:

```bash
build/estimate --check data/delay_estimate_check.csv
```

Measure the learned model on a labeled subset:

```bash
build/estimate --model-benchmark data/delay_estimate_ans.csv \
  --offset 900000 --limit 100000
```

`--benchmark` runs the exact router and is intended only for small diagnostic samples.

## Offline Exact Labeling

The exact router can label additional endpoint pairs for offline experiments:

```bash
build/estimate --exact-label \
  -in refine-logs/candidates.csv \
  -out refine-logs/exact_answers.csv \
  --limit 200000
```

The input uses the normal request CSV schema and the output uses the normal answer schema.
`--limit` is optional. This mode is deliberately separate from the contest `-in/-out` path: exact
routing is suitable for generating diagnostic labels, but is far too slow for 100 million queries.
For resumable multi-process labeling, use `scripts/label_candidates.py`.

## Generalization Experiments

The public one-million-row set does not currently show a large memorization gap: the original
900k/100k train/validation scores are 94.5582 and 94.2958. A stricter 800k/100k/100k protocol was
therefore used to keep development and final evaluation separate. Its final-test learning curve is:

```text
training rows     final-test score
100,000           93.8623
250,000           94.0404
500,000           94.1589
800,000           94.2343
```

An additional 200k short-path queries were generated only from the first 800k rows and labeled by
the exact oracle. After selecting their training weight on the middle 100k rows, the score on the
untouched final 100k changed from 94.2346 to 94.2433. This 0.0087 gain is not large enough to justify
blindly scaling exact labeling. The current bottleneck is short-path topology representation rather
than the overall number of training rows, so the production model is intentionally unchanged.

Full methodology, leakage exclusions, grouped errors, and artifact paths are recorded in
`refine-logs/EXPERIMENT_RESULTS.md`.

The confirmed organizer constraints, reproducible packaging preflight, locked production choice,
and unresolved final-archive questions are tracked in `docs/SUBMISSION_CHECKLIST.md`.

## Model Reproduction

Training uses LightGBM only offline. The executable itself has no Python or LightGBM dependency.

To reproduce the offline training environment, create a virtual environment and install
`requirements-experiments.txt`:

```bash
python3 -m venv .venv-experiments
source .venv-experiments/bin/activate
python -m pip install -r requirements-experiments.txt
```

```bash
python3 scripts/train_model.py \
  --trees 128 --leaves 127 --depth 8 --learning-rate 0.08 \
  --output models/delay_model_v2_128.txt

python3 scripts/export_lightgbm_cpp.py \
  models/delay_model_v2_128.txt src/generated_model.hpp
```

The exporter flattens all trees into constant C++ arrays and turns categorical splits into bitsets.
This preserves LightGBM predictions while removing runtime model parsing and external libraries.

## Project Layout

```text
arch/                       Official architecture data
data/                       Public golden/check data
docs/RESULTS.md             Reproducible accuracy and performance record
models/delay_model_v2_128.* Final offline model and training metadata
scripts/                    Training and C++ export tools
src/architecture.*          JSON loading, validation, Gap/Net semantics
src/router.*                Exact implicit-graph A* oracle
src/exact_csv.*             Offline batch interface to the exact oracle
src/ml_estimator.*          Production feature extraction and inference
src/generated_model.hpp     Generated self-contained tree arrays
tests/                      Architecture, exact router, model, and CLI tests
refine-logs/                Tracked experiment summaries; raw generated artifacts stay local
```

The current mode comparison, reproducibility notes, and repository boundaries are summarized in
`docs/REPOSITORY_STATUS.md`.

Build directories, local toolchains, virtual environments, historical candidate models, and raw
experiment outputs are intentionally excluded by `.gitignore`. They are not required to build or
run the production estimator.
