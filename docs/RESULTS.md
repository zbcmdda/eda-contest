# Experiment Results

All measurements below were produced on 2026-08-15 with GCC 13.3 in Release mode. The public
answer file contains 1,000,000 rows. Rows 0-899,999 train the model; rows 900,000-999,999 are a
strict validation set.

## Correctness Oracle

The exact implicit-graph router models 208 input-port states per physical SRB. It expands Arc
templates on demand and resolves Net destinations using the official rules:

- Line delays are accumulated for every crossed row/column boundary.
- A crossable Block does not count toward Net span and contributes its cross delay once.
- A non-crossable Block counts toward span; a Net is disconnected if its final SRB is occupied.

All eight official check Paths have valid edges, their edge delays sum to the published minimum,
and independent A* returns the same minimum for every row.

## Final Model

Configuration:

```text
objective:          log1p(delay), squared error
trees:              128
maximum depth:      8
maximum leaves:     127
learning rate:      0.08
training rows:      900,000
validation rows:    100,000
features:           51
```

Validation metrics after integer rounding in C++:

```text
official accuracy score:    94.2958 / 100
mean relative error:         1.4674%
within 5% relative error:   95.549%
within 10% relative error:  98.881%
within 20% relative error:  99.798%
```

The dominant feature is `relaxed_after_source`: it applies a source `Z*` Net exactly, then uses a
channel-relaxed one-dimensional shortest-path table for the remaining signed x/y displacement.
Port categories, Line/Block features, endpoint semantics, and target Arc minima correct the
remaining architecture-specific residual.

## Candidate Selection

```text
model                    validation score    single-thread query/s
v1 depth-8, 128 trees        93.2395              not retained
v1 depth-10, 256 trees       94.3839              not retained
v2 depth-8, 128 trees        94.2958              169,264
v2 depth-10, 128 trees       94.4837               64,443
v2 MAPE depth-8              93.6887              not retained
v2 log-L1 depth-8            93.5223              not retained
```

The contest time score is linear: `(1 - avg_time / 1800) * 100`. The depth-10 candidate gains only
0.19 accuracy points but loses far more weighted time score, so depth 8 is the final choice.

## Performance

Production command on a one-million-row request, including CSV parsing and result writing. Five
consecutive runs produced:

```text
real times:             6.02, 5.99, 5.98, 6.06, 5.95 s
mean real time:         6.00 s
peak resident memory:  45.9 MB
output rows per run:    1,000,001 including header
```

The public-set target is 18 seconds. Linear extrapolation to 100 million rows is about 602 seconds,
leaving substantial margin under the 1,800-second hard limit. Actual evaluation storage and CPU
may differ, so the full-scale extrapolation remains a risk estimate rather than a guarantee.

All five one-million-row runs produced the same SHA-256 result:

```text
4eab31df19e402fc64ac804d83c5875d613619bb35a52f3f4a522f4e2c1fd56c
```

## Hot-path audit (2026-08-22)

The production model and feature set were kept unchanged. The C++ path was
optimized in three semantics-preserving places: endpoint parsing now consumes
`string_view` slices and uses a collision-checked short-name hash; port-name
semantics and generated categorical codes are precomputed once per estimator;
and crossable Net Blocks are traversed by interval jumps rather than testing
every physical coordinate. Result formatting uses a reusable buffer and
`to_chars`.

On the same one-million-row request derived from the public answer set, five
consecutive runs after the change took `5.29, 5.29, 5.32, 5.28, 5.27 s`
(mean `5.29 s`, peak RSS `45.9 MB`). All output hashes remained
`4eab31df19e402fc64ac804d83c5875d613619bb35a52f3f4a522f4e2c1fd56c`, and the
strict 100,000-row model benchmark remained `94.2958` with mean relative error
`1.4674%`. The benchmark itself took `0.518 s` for 100,000 rows. CTest
remained 5/5; exact routing and the default no-flag path were not changed.

Using the historical linear 100-million-row extrapolation, `5.29 s` per
million implies about `529 s`, versus the previous `600 s`. With accuracy and
consistency unchanged, this is an indicative weighted-score increase of about
`+0.59` points (hidden input distribution and hardware can differ).

Using the strict validation accuracy and linear runtime extrapolation gives an indicative weighted
score near 90.4: 75.44 accuracy points, about 10.0 time points, and 5 consistency points. This is
not a substitute for the organizer's hidden-set measurement.

## Tree-prefix and LTO audit (2026-08-22)

The generated predictor now accepts an experimental `tree_limit` prefix. The
public CLI exposes it as `--model-trees N`, defaulting to 128; valid values are
1 through 128. The exporter emits the same interface, and the estimator rejects
out-of-range limits. The default executable and explicit `--model-trees 128`
fixture outputs compare byte-for-byte; the five one-million-row default hashes
remain unchanged. This option is retained for measurement only and is not a
submission default change.

The frozen final 100,000 rows (`offset=900000`) were not used to choose a tree
limit. Prefix screening used the development block (`offset=800000,
limit=100000`):

```text
trees   acc_score  elapsed_s/100k  delta_vs_128  proxy_score
64      93.7207       0.304447       -0.8452       92.5766
80      94.2157       0.373945       -0.3502       92.2392
96      94.3585       0.430323       -0.2074       91.7968
112     94.4465       0.505347       -0.1194       91.3772
128     94.5659       0.559812        0.0000       91.0011
```

As a contiguous-block robustness audit inside the first 900,000 rows, the
score deltas against the 128-tree prefix were:

```text
held-out offset       64        80        96        112
600000             -0.8250   -0.3442   -0.2049   -0.1186
700000             -0.8335   -0.3539   -0.2109   -0.1214
800000             -0.8452   -0.3502   -0.2074   -0.1194
mean               -0.8346   -0.3494   -0.2077   -0.1198
worst               -0.8452   -0.3539   -0.2109   -0.1214
```

The contiguous blocks show a consistent *accuracy* decrease for every shorter
prefix, but that is not a rejection under the contest objective: accuracy is
only 80% of the score and runtime is also scored.  Applying the published
score formula to the five-run full-CLI means below (linear 100-million-row
extrapolation, with 5 consistency points) gives the following development
proxies:

```text
trees   proxy_score   delta_vs_128
64        92.5766       +1.5755
80        92.2392       +1.2382
96        91.7968       +0.7957
112       91.3772       +0.3761
128       91.0011        0.0000
```

For example, the 64-tree result is
`0.80 * 93.7207 + 0.15 * (1 - 288.0 / 1800) * 100 + 5 = 92.5766`.
All tested prefixes have CV below 2% and save more than 0.15 s/million
relative to 128 trees.  Therefore 64 trees is the current **development-set
composite-score candidate**, not a rejected candidate.  It has not been
evaluated on the frozen final block (`offset=900000`), so this is not a final
production-model conclusion: the submission default remains 128 and no tree
count is hard-coded as the new default.

For completeness, five one-million-row runs per prefix were interleaved in
alternating tree-count order. CV is population standard deviation divided by
the mean; RSS is the maximum observed value:

```text
trees  times_s                  mean_s  CV       RSS_KB  SHA-256 (all 5 runs)
64     2.93,2.88,2.88,2.87,2.84  2.880   1.01%    45020  2a26fc...6fedb1d
80     3.75,3.74,3.87,3.73,3.71  3.760   1.50%    45020  d14620...f1876d4
96     4.41,4.33,4.47,4.37,4.56  4.428   1.82%    45360  3d166f...fbc1470
112    4.99,5.00,5.09,4.92,5.08  5.016   1.25%    45744  fd5e38...f846024
128    5.53,5.67,5.59,5.60,5.52  5.582   0.97%    46000  4eab31...1fd56c
```

An independent LTO/IPO control was also built in `build-lto`. Both builds
passed all 5 CTest cases. Five alternating one-million-row pairs produced:

```text
build       times_s                  mean_s  CV       RSS_KB  SHA
Release     5.70,5.58,5.46,5.52,5.55  5.562   1.43%    45872  4eab31...1fd56c
Release+IPO 5.39,5.43,5.46,5.49,5.40  5.434   0.68%    45872  4eab31...1fd56c
```

The IPO build is only about `2.30%` faster, below the required `3%` gain, so
it is not adopted and the regular Release build remains the production build.
No new final-set accuracy claim is made in this audit.

## Bounded medium-short residual candidate (2026-08-24)

The earlier direct short specialists failed because replacing the production prediction discarded
useful global structure. The successful candidate instead predicts a small log-ratio correction on
top of the rounded production delay. It preserves exact Arc/Net/Block/Line semantics in a bounded
beam descriptor with fixed arrays and deterministic tie breaks:

```text
Manhattan 0--32:   three hops, top-4 then top-2 beam, 64 trees depth 12
Manhattan 33--128: two hops, top-4 beam,              16 trees depth 14
Manhattan >128:    unchanged production model
residual formula:  round((base_delay + 1) * exp(1.5 * residual) - 1)
objective:         L1 on log((golden + 1) / (base_delay + 1))
```

Rows 0--799,999 supplied residual labels and rows 800,000--899,999 were development only. Five
independent seeds improved the complete development block by `+0.4005` to `+0.4299` (mean
`+0.4129`); all four contiguous 25k blocks and all four distance buckets improved for every seed.
After the configuration, threshold, multiplier, and implementation were locked, the final 100k was
evaluated once:

```text
metric                         baseline       candidate       delta
official score                 94.2958        94.7564         +0.4605
mean relative error             1.4674%        1.3370%        -0.1304 pp
within 5%                      95.549%        96.390%         +0.841 pp
within 10%                     98.881%        99.166%         +0.285 pp
within 20%                     99.798%        99.876%         +0.078 pp
```

Final 100k distance scores:

```text
distance       rows       baseline       candidate       delta
0--16           806        64.3008         71.4785        +7.1777
17--32         2231        76.2246         79.9225        +3.6979
33--64         7202        87.0185         88.7251        +1.7066
65--128       19666        92.4737         93.4768        +1.0031
>128          70095        96.4749         96.4749         0.0000
```

The C++ and Python three-hop features matched in all cells for 1,024 checked rows. ASan/UBSan and
all six CTest cases pass. Five one-million-row candidate runs produced the same SHA-256:

```text
a8581a9fcf047e902815a59849545c348f732d797e46a7149b4be3b82b6cb10f
```

On the local Zig/Clang build, threshold 128 took `15.08, 14.67, 14.90, 14.75, 14.74 s` per million
(mean `14.83 s`, peak RSS about 35 MiB). This is below the public 18-second guideline and linearly
extrapolates to about 1,483 seconds, but must still be measured on the actual 100-million input and
target compiler. The same-build accuracy/time Pareto is:

```text
threshold       development delta       final delta       seconds / 1M
32                    +0.1127              +0.1404             10.92
64                    +0.2159              +0.2633             12.25
96                    +0.3266              +0.3737             13.28
128                   +0.4048              +0.4605             14.83
baseline                0                    0                 about 10.35
```

The candidate remains behind an explicit `--short-residual` flag. Accuracy and the hard runtime
limit are qualified locally, but the no-flag 128-tree model remains the default until a real
100-million-row/target-machine weighted-score comparison selects a submission policy.

## Architecture-driven short router (2026-08-24, strongest accuracy candidate)

The learned residual above showed that short topology matters, but it remained far below the desired
segment accuracy. The replacement does not learn the short delay. For Manhattan distance `0--32`, it
runs weighted A* directly on the official Arc/Net graph; longer queries retain the production model.
Its fixed implementation is exposed as:

```bash
build-local/estimate -in request.csv -out result.csv --short-route
```

The speedup over the generic exact router comes from four architecture-derived changes:

- a coordinate-free all-pairs input-port shortest-path lower bound, combined with the existing
  spatial relaxed lower bound;
- terminal Arcs compiled into one dense lookup so the search scans only routing Arcs;
- exact Net transitions compiled by axis and Block-gap band class;
- weighted A* with fixed weight `1.2`, followed by fixed calibration multipliers `0.999000` for
  distance `0--16` and `0.997750` for `17--32`.

The weight was selected on rows 800,000--899,999. Calibration used only rows 0--799,999. After the
weight, thresholds, and multipliers were recorded in
`refine-logs/short_route_qualification_v1/results.json`, rows 900,000--999,999 were evaluated once;
there was no post-final tuning and sealed labels were never read.

Development qualification:

```text
distance       rows       baseline       short route       within 5%       within 10%
0--16           819        67.4704          97.1461           98.657%          100.000%
17--32         2136        78.3906          97.0535           99.251%          100.000%
```

Every contiguous 25k development block scores at least `96.8341` in `0--16` and `96.8563` in
`17--32`. The complete development score improves from `94.5659` to `95.2076` (`+0.6417`).

Locked final 100k:

```text
metric                         baseline       short route       delta
official score                 94.2958          95.0229         +0.7271
mean relative error             1.4674%          1.2550%        -0.2124 pp
within 5%                      95.549%          97.153%         +1.604 pp
within 10%                     98.881%          99.657%         +0.776 pp
within 20%                     99.798%          99.991%         +0.193 pp
```

Final short segments:

```text
distance       rows       baseline       short route       delta       within 5% / 10%
0--16           806        64.3008          96.8621        +32.5613       98.635% / 100%
17--32         2231        76.2246          97.0510        +20.8264       99.328% / 100%
0--32          3037        73.0601          97.0009        +23.9408       99.144% / 100%
```

Five complete one-million-row runs took `16.23, 16.02, 16.14, 16.03, 16.37 s` (mean `16.16 s`),
with peak RSS about `137 MiB`. All five outputs have SHA-256:

```text
63aace61548f432e166b60419f3f142fa2d6f451427d2bcc896b96b888cadbcf
```

The same-build no-flag control took about `9.84 s`. Linear candidate extrapolation is about
`1,616 s` per 100 million rows, below the 1,800-second hard limit but still requiring confirmation
on the real request distribution and target machine. Exact-router regression over all 29,293 public
queries through distance 32 is byte-identical before and after the optimization. Release and
ASan/UBSan builds both pass all six tests. `--short-route` is therefore the strongest qualified
accuracy candidate; the no-flag output remains unchanged until the real submission policy is chosen.
