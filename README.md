# GTA VI launch-day Twitch viewers — hour-by-hour forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/wahanex2/gta6-twitch-ccu-forecast-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/wahanex2/gta6-twitch-ccu-forecast-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast Grand Theft Auto VI's Twitch concurrent viewers for each hour
2–24 of release day (2026-11-19 UTC), as five quantiles (5/25/50/75/95%),
from the hour-1 print and the comparables the case ships.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (else `/output`).

The key observation from the comparables (38–42 past big game launches) is
that the launch-day curve depends strongly on the launch's size: the
biggest launches are already near their ceiling at hour 1 and end the day
near 1× it, while mid-size launches rise several-fold into the US evening,
and small launches whose unlock falls later in the UTC day explode from a
near-zero hour 1. So the model conditions on scale:

1. Per UTC hour h = 2..24, take each comparable's log-ratio
   `ln(viewers_h / viewers_hour1)`.
2. Fit that ratio as a quadratic in `log10(hour-1 viewers)` and take the
   empirical quantiles of the fit residuals — a scale-conditioned median
   curve plus an honest spread that widens with horizon.
3. Independently take Gaussian-kernel weighted quantiles of the same
   ratios (bandwidth 0.5 dex), centred on the case's own hour-1 print.
4. Average the two sets of log-quantile offsets and apply them to the
   case's hour-1 print (clamped to the comparables' observed hour-1 range,
   so the quadratic never extrapolates past the biggest launch seen).
5. If too few comparables are usable, fall back to the house baseline
   (persistence with a 0.15/√hour log spread).

The case's own pre-release history is not used: in leave-one-comparable-out
validation a pre-release drop feature did not improve the loss.

## Validation

Leave-one-comparable-out over the 37 complete comparables in the kit's
practice cases (mean pinball loss on the log scale, lower is better):

| model | all comparables | big launches (hour-1 ≥ 50k) |
|---|---|---|
| house baseline (persistence) | 0.930 | 0.251 |
| this model | 0.379 | 0.205 |

Practice answers are not in the kit, so no practice score is claimed; the
numbers above are the comparable-based validation only.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
