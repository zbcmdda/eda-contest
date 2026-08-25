# Submission Checklist

This checklist records only requirements confirmed by the official problem PDF and Q&A, plus
local preflight checks. It deliberately does not invent an archive layout that the organizer has
not specified.

## Current algorithm policy

- The strongest qualified accuracy candidate is `--short-route` (`95.0229` on the locked public
  final block, `16.16 s/1M` local mean).
- The no-flag 128-tree model remains the conservative baseline (`94.2958` on the same block) and
  is kept unchanged until a same-machine, full-input weighted-score comparison selects the final
  submission policy.
- Build the executable from the tracked source before comparing policies. Existing `build/` and
  `build-local/` directories are ignored local artifacts, not repository contents.
- The earlier `--short-residual[-threshold N]` Pareto candidate remains available for comparison.

## Confirmed organizer requirements

- Entrypoint invocation:

  ```bash
  estimate -in ./delay_estimate_request.csv -out ./delay_estimate_result.csv
  ```

- The hidden input contains 100 million reachable requests.
- The complete run, including architecture loading, preprocessing, computation, and output, must
  finish within 1,800 seconds.
- Peak memory must not exceed 2 GiB.
- Only one CPU thread is allowed; GPU and multithreaded execution are forbidden.
- The evaluator runs the program five times and requires byte-identical results for full
  consistency credit.

## Reproducible staging preflight

Run from the project directory:

```bash
source .local-toolchain/activate
cmake --build build-local -j 4
ctest --test-dir build-local --output-on-failure
build-local/estimate --model-benchmark data/delay_estimate_ans.csv \
  --offset 900000 --limit 100000

preflight_dir="$(mktemp -d /tmp/pango-submission-preflight.XXXXXX)"
cmake --install build --prefix "$preflight_dir/package"
cd "$preflight_dir"
package/estimate \
  -in '/home/zhangce/eda contest/EDA-PANGO-SRB-Arch-Delay/tests/fixtures/request.csv' \
  -out result.csv
cmp result.csv \
  '/home/zhangce/eda contest/EDA-PANGO-SRB-Arch-Delay/tests/fixtures/result.csv'
sha256sum package/estimate result.csv
```

The install command intentionally stages a directory but does not create the final archive. The
archive type, top-level directory name, and permitted companion files remain organizer-controlled.

## Current qualified values

As of 2026-08-24:

```text
build/estimate SHA-256:
c2c92d53d1ab71d1066774220ca20cf1b6523bcda79069dd5d89fac9317f0538

strict final 100k public score: 94.2958
historical 1M mean runtime:     5.29 s after hot-path optimization
historical peak RSS:           about 45.9 MiB
historical 1M output SHA-256:
4eab31df19e402fc64ac804d83c5875d613619bb35a52f3f4a522f4e2c1fd56c
```

The current `cmake --install build` staging tree contains `estimate` and the five JSON files under
`arch/`, and is about 7.4 MiB. Running that tree from outside the source directory matches the CLI
fixture byte-for-byte.

The current `cmake --install build-local` tree also runs `--short-route` from an unrelated working
directory and matches `tests/fixtures/short_route_result.csv` byte-for-byte. Its unstripped local
Zig/Clang staging size is about 19 MiB; this is a development preflight, not the final target build.

## Must resolve before the final archive

1. Obtain the organizer's complete submission/packaging rules: the current PDF says to submit an
   executable but does not say whether the five architecture JSON files may or must accompany it,
   nor which archive layout is accepted.
2. If the real 100-million-row request or target evaluation machine becomes available, run both the
   pure 128-tree model and `--short-route` under identical conditions. Use the short-residual
   thresholds 32/64/96/128 only as secondary Pareto controls; `--exact-threshold 16` is retained as
   an older comparison point. Select a non-default path only from measured total score, runtime,
   memory, and five-run determinism.
3. Confirm the evaluator's Linux distribution and C++ runtime compatibility, or qualify the exact
   target container. The historical binary is dynamically linked against glibc, libstdc++, libm,
   and libgcc_s.
4. After any intentional source/model/binary change, rerun all six tests, the strict 100k benchmark,
   at least one full public 1M CLI run, and the five-run consistency check before updating hashes.
