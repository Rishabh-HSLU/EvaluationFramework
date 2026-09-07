# Missing-bar structure of the curated real panel

Source: `/home/rishabh/PycharmProjects/EvaluationFramework/data/curated/real_prices.parquet` — 53,850 rows x 600 tickers.
Restricted to 137 full-length 390-bar sessions; 2 early-close sessions excluded by bar count (2019-11-29, 2019-12-24).

## 1. Overall shape

- (ticker, session) pairs: **82,200** (600 tickers x 137 sessions)
- Fully complete (zero missing bars): **5,424** (6.6%)
- At least one missing bar: **76,776** (93.4%)

Missing-bar counts among the incomplete pairs (out of 390 bars):

| statistic | missing bars |
|---|---|
| count | 76,776 |
| min | 1 |
| p25 | 7 |
| median | 25 |
| p75 | 62 |
| p90 | 101 |
| p95 | 124 |
| p99 | 165 |
| max | 390 |
| mean | 39.66 |

## 2. Concentration across tickers

- Tickers with at least one incomplete session: **600** of 600 (**100.0%**)
- Tickers with **no** complete session at all: **216** (36.0%)
- Worst decile by incomplete-session rate (216 tickers, rate >= 100.0%) accounts for **38.5%** of all incomplete (ticker, session) pairs
- Worst five tickers: WSM (100%), AAN (100%), WTFC (100%), AAXN (100%), ABCB (100%)

Per-ticker share of sessions that are incomplete:

| statistic | incomplete sessions (%) |
|---|---|
| count | 600 |
| min | 9 |
| p25 | 95 |
| median | 99 |
| p75 | 100 |
| p90 | 100 |
| p95 | 100 |
| p99 | 100 |
| max | 100 |
| mean | 93.40 |

## 3. Contiguity of gaps within a session

- Contiguous missing-bar runs: **2,190,311**, covering 3,044,819 missing bars
- Isolated single minutes (run length 1): **1,672,194** (**76.3%** of runs, 54.9% of missing bars)
- Runs of 10+ consecutive minutes: **3,507** (0.16% of runs)

Run-length distribution:

| statistic | run length (minutes) |
|---|---|
| count | 2,190,311 |
| min | 1 |
| p25 | 1 |
| median | 1 |
| p75 | 1 |
| p90 | 2 |
| p95 | 3 |
| p99 | 5 |
| max | 390 |
| mean | 1.39 |

| run length | runs | share of runs | missing bars | share of bars |
|---|---|---|---|---|
| 1 | 1,672,194 | 76.3% | 1,672,194 | 54.9% |
| 2 | 350,892 | 16.0% | 701,784 | 23.0% |
| 3-5 | 152,941 | 7.0% | 522,861 | 17.2% |
| 6-10 | 11,153 | 0.5% | 76,224 | 2.5% |
| 11-30 | 2,879 | 0.1% | 45,498 | 1.5% |
| 31-60 | 133 | 0.0% | 5,073 | 0.2% |
| 61-390 | 119 | 0.0% | 21,185 | 0.7% |

## 4. Time-of-day location

- Missing rate per minute position, averaged over all 82,200 pairs
- First 30 minutes: **13.18%** | middle: **9.83%** | last 30 excluding the close bar: **0.95%**
- Ratio open/middle: **1.34x**, close/middle: **0.10x**
- Excluding position 390, the pattern is a **dirty open decaying into a clean close**: missingness peaks in the first minutes, drifts down through the session, and all but vanishes over the final half hour.

**Position 390 (16:00) is a structural outlier and is excluded from the figures above.**

- Missing rate at position 390: **47.66%**
- Missing rate at position 389 (15:59): **0.05%** — the close bar is **911x** its neighbour
- Per-session spread of the close-bar gap: min **39.2%**, median **48.2%**, max **53.5%** across all 137 sessions

It affects roughly half of all tickers in *every* session rather than clustering on particular days, so it is a systematic property of the 16:00 print — plausibly closing-auction timing — not an outage. Note `session_clock.py` already documents a separate close-bar issue (`SESSION_OFFSET` excluding minute position 390 from FFF slots).

Ten worst minute positions excluding 390:

| minute position | missing rate |
|---|---|
| 1 | 25.69% |
| 2 | 20.36% |
| 3 | 16.95% |
| 4 | 15.68% |
| 7 | 15.32% |
| 9 | 14.81% |
| 8 | 14.79% |
| 6 | 14.67% |
| 14 | 14.63% |
| 13 | 14.63% |

## 5. Relationship to the 70% inclusion floor

Coverage is the per-ticker non-NaN fraction over the whole market clock — the quantity `CurationPipeline` thresholds at `COVERAGE_FLOOR = 70%`.

- Coverage range across retained tickers: **70.2%** to **99.9%** (median 92.3%)

| group | tickers | mean coverage | mean complete-session rate | share of all incomplete pairs |
|---|---|---|---|---|
| near floor (70%-80%) | 74 | 76.2% | 0.1% | 13.2% |
| well covered (>= 95%) | 233 | 98.0% | 15.3% | 35.2% |
| all retained tickers | 600 | 90.5% | 6.6% | 100.0% |

- Correlation between overall coverage and session-completeness rate: **+0.472**
