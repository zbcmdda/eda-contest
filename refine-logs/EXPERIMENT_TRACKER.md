# Experiment Tracker

| Run ID | Milestone | Purpose | Variant | Split | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| R001 | M0 | Train/validation baseline audit | production C++ model | 900k/100k | MUST | DONE | Train 94.5582, validation 94.2958 |
| R002 | M0 | Dataset overlap and coverage | public data | 900k/100k | MUST | DONE | Zero exact-pair overlap; distributions closely match |
| R003 | M0 | Exact oracle public sanity | exact A* | 10 stratified rows | MUST | DONE | 0 mismatches; 1.39 query/s for mixed long rows |
| R010 | M1 | Learning curve | direct log target | 80/10/10 split | MUST | DONE | 100k 93.8623, 250k 94.0404, 500k 94.1589, 800k 94.2343 on final test |
| R011 | M1 | Grouped robustness | production model | distance/port/region/gap | MUST | DONE | Short 0-16 score 64.30; Block groups about 90; long paths about 95+ |
| R020 | M2 | Candidate generation | short translated templates | train-only pool | MUST | DONE | 200k candidates, deterministic, no public pair overlap |
| R021 | M2 | First exact label round | translated short templates | 200k | MUST | DONE | 200k/200k reachable; 8 workers, about 41 s |
| R030 | M3 | Target ablation | residual, specialist, weighting | strict 80/10/10 | MUST | DONE | Residual and weighting no gain; augmented specialist test 94.2433 vs 94.2346 |
| R040 | M4 | Production qualification | existing production model | regression and CLI | MUST | DONE | Baseline retained; no candidate cleared the score/time gate |
| R050 | M5 | Medium-short residual search | A0/A1/A2, direct vs residual | 800k/100k development | MUST | DONE | Two-hop L1 residual produced stable positive signal |
| R051 | M5 | Compact bounded candidate | depth-3 <=32, depth-2 33--128 | five seeds, contiguous blocks | MUST | DONE | Full development +0.4005 to +0.4299; every segment positive |
| R052 | M5 | Locked final qualification | optional `--short-residual` C++ | frozen final 100k | MUST | DONE | 94.2958 -> 94.7564; 1M five-run SHA stable; 14.83 s local mean |
| R053 | M6 | Short exact-search decomposition | exact and weighted A* by 0--16/17--32 | rows 0--899999 | MUST | DONE | Exact matches golden; port-compatible lower bound and compiled Net transitions cut search time |
| R054 | M6 | Locked short-route qualification | `--short-route`, weight 1.2, train-only calibration | 800k/100k development | MUST | DONE | 0--16=97.1461; 17--32=97.0535; every 25k block above 96.8 |
| R055 | M6 | Once-opened final and performance | locked `--short-route` C++ | frozen final 100k + public 1M | MUST | DONE | overall 95.0229; segments 96.8621/97.0510; 1M mean 16.16 s; five-run SHA stable |
