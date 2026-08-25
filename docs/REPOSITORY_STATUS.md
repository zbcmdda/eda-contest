# Repository Status

Updated 2026-08-25.

## Runtime policies

| Policy | Command | Locked public final score | Local 1M runtime | Role |
| --- | --- | ---: | ---: | --- |
| Conservative baseline | no extra flag | 94.2958 | about 9.84 s | Stable control and current no-flag behavior |
| Latest accuracy candidate | `--short-route` | 95.0229 | 16.16 s | Strongest qualified accuracy path |
| Earlier residual candidate | `--short-residual` | 94.7564 | 14.83 s | Historical Pareto comparison |

The scores are from the locked public final block, not the hidden 100-million-row evaluation. The
`--short-route` path improves short-query accuracy by running weighted A* over the official
architecture graph for Manhattan distance `0--32`; longer queries use the estimator model.

## Rebuild and run the latest candidate

Use a fresh build directory when checking out the repository:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
build/estimate -in request.csv -out result.csv --short-route
```

The input and output paths must be different. Without `--short-route`, the executable runs the
conservative baseline by design. The final competition policy still requires a full-input,
target-machine comparison of accuracy, runtime, memory, and five-run byte-identical output.

## What the repository tracks

Tracked content includes the C++ source, generated self-contained model headers, architecture JSON
files, public validation data, offline scripts, tests, final model metadata, and concise experiment
reports.

The following remain local and are intentionally ignored by `.gitignore`:

- CMake/Ninja build directories (`build*`);
- `.venv-experiments` and `.local-toolchain`;
- raw generated files under `refine-logs/`;
- historical candidate models under `models/candidates/`.

These artifacts are not required to rebuild or run either tracked runtime policy.
