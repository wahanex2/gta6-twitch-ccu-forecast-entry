"""gta6-twitch-ccu-forecast-01 entry: scale-conditioned comparable-curve forecaster.

Model
-----
Launch-day Twitch viewership does not stay at hour 1's print: for a rolling
midnight-local release the UTC day starts with only New Zealand / Asia /
Europe unlocked, dips through the overnight lull, then climbs as the US
unlocks (05:00Z east coast, 08:00Z west coast) into the US evening peak.
How far it climbs depends strongly on the launch's size: the biggest
launches (hour-1 print in the hundreds of thousands) are already near their
ceiling at hour 1 and end the day near 1x, while mid-size launches rise
several-fold. Small launches whose unlock falls later in the UTC day explode
from a near-zero hour 1. So this forecaster conditions on scale:

1. From the case's comparables (past big game launches, each with hourly
   Twitch counts from 14 days before to 3 days after its release day) build,
   per UTC hour h = 2..24, the log-ratio r = ln(viewers_h / viewers_hour1).
2. Per hour, fit r as a quadratic in x = log10(hour-1 viewers) across the
   comparables, and take the empirical quantiles of the fit residuals.
   That gives a scale-conditioned median curve plus an honest residual
   spread that widens with horizon (residual dispersion grows over the day).
3. Independently, per hour, take Gaussian-kernel weighted quantiles of r
   (bandwidth 0.5 dex in log10 hour-1 viewers) centred on the case's own
   hour-1 print. This local estimate catches structure the quadratic misses.
4. Average the two sets of log-quantile offsets (a 50/50 blend) and apply
   them to the case's hour-1 print. The case's own pre-release history is
   not used: in leave-one-comparable-out validation a pre-release drop
   feature did not improve the loss.
5. Fallbacks: if too few comparables are usable, fall back to the house
   baseline (persistence with a 0.15/sqrt-hour log spread). The hour-1 print
   used for scaling is clamped to the comparables' observed hour-1 range so
   the quadratic never extrapolates wildly beyond the biggest launch seen.
   And if a case file itself cannot be modelled at all (missing fields, an
   hour-1 print that is not a number), main() still writes a valid
   persistence forecast for it instead of crashing: a missing output scores
   as if every quantile equalled the hour-1 print and an entry with any
   invalid case takes no prize, so no single bad case may void the run.

Validation (leave-one-comparable-out over the 37 complete comparables in
the kit's practice cases, scoring mean pinball loss on the log scale):
  house baseline (persistence):        0.930 overall, 0.251 for big
                                        launches (hour-1 >= 50k)
  this model:                          0.379 overall, 0.205 for big
                                        launches
Practice answers are not in the kit, so no practice score is claimed here;
the numbers above are the comparable-based validation only.

How the grader runs it: `python3 -P -s /bot/main.py` (Python 3.13,
standard library only, no network). -P keeps this file's directory OFF
sys.path, so the sys.path line below puts it back first (kept for parity
with the starter, though this entry is a single file). Reads every <id>.in
in $MAKEFUN_INPUT_DIR (else /input) and writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output): {"hours": [{"hour": h, "q": [q05, q25,
q50, q75, q95]}...]} for hours 2..24, each q a finite number in (0, 1e9],
non-decreasing, deterministic (no randomness anywhere).
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles at those levels (used only by the fallback).
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Log spread of the fallback persistence forecast, per sqrt hour.
FALLBACK_SIGMA = 0.15
# Gaussian-kernel bandwidth (dex of log10 hour-1 viewers) for the local fit.
KERNEL_BW = 0.5
# Minimum comparables with both hour 1 and hour h to trust the fitted model.
MIN_COMPS = 8


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as the starter rounds it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def solve_linear(matrix, vector):
    """Small dense solve by Gaussian elimination with partial pivoting."""
    n = len(vector)
    rows = [row[:] + [v] for row, v in zip(matrix, vector)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(rows[r][i]))
        if abs(rows[piv][i]) < 1e-12:
            raise ValueError("singular system")
        rows[i], rows[piv] = rows[piv], rows[i]
        for r in range(i + 1, n):
            factor = rows[r][i] / rows[i][i]
            for c in range(i, n + 1):
                rows[r][c] -= factor * rows[i][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (rows[i][n] - sum(rows[i][j] * out[j] for j in range(i + 1, n))) / rows[i][i]
    return out


def empirical_quantile(sorted_values, tau):
    """Linear-interpolation quantile of an already sorted list."""
    if not sorted_values:
        raise ValueError("no values")
    pos = tau * (len(sorted_values) - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def comparable_day_series(comp):
    """Release-day hourly viewers {hour 1..24: viewers} for one comparable.

    Hours are counted from the comparable's day_start (00:00 UTC of its
    release date). Returns None when hour 1 is missing or non-positive.
    """
    day_start = parse_time(comp["day_start"])
    by_hour = {}
    for row in comp.get("hourly") or []:
        viewers = row.get("viewers")
        if not viewers or viewers <= 0:
            continue
        hour = int((parse_time(row["from"]) - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            by_hour[hour] = viewers
    if 1 not in by_hour:
        return None
    return by_hour


def quadratic_fit(xs, ys):
    """Least-squares quadratic in (x - 5); returns coefficients [c0, c1, c2]."""
    design = [[1.0, x - 5.0, (x - 5.0) ** 2] for x in xs]
    dim = 3
    normal = [[sum(r[i] * r[j] for r in design) for j in range(dim)] for i in range(dim)]
    rhs = [sum(r[i] * y for r, y in zip(design, ys)) for i in range(dim)]
    # Tiny ridge for numerical stability; far too small to move the fit.
    for i in range(dim):
        normal[i][i] += 1e-9
    return solve_linear(normal, rhs)


def quadratic_value(coef, x):
    xc = x - 5.0
    return coef[0] + coef[1] * xc + coef[2] * xc * xc


def weighted_quantile(pairs, tau):
    """Weighted quantile; pairs is a list of (value, weight) sorted by value."""
    total = sum(w for _, w in pairs)
    if total <= 0:
        raise ValueError("no weight")
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= tau * total:
            return value
    return pairs[-1][0]


def hour_log_quantiles(series_list, x_case, hour):
    """Blend of quadratic-fit and kernel-local log-ratio quantiles for one hour.

    series_list: [(log10 hour-1 viewers, {hour: viewers})] for comparables
    that have both hour 1 and this hour. x_case: log10 of the case's hour-1
    print (already clamped to the observed range). Returns the five
    log-quantile offsets at LEVELS, or None when data is too thin.
    """
    xs = []
    ys = []
    for x, by_hour in series_list:
        if hour in by_hour:
            xs.append(x)
            ys.append(math.log(by_hour[hour] / by_hour[1]))
    if len(xs) < MIN_COMPS:
        return None
    # --- quadratic fit + empirical residual quantiles -------------------
    try:
        coef = quadratic_fit(xs, ys)
    except ValueError:
        return None
    base = quadratic_value(coef, x_case)
    residuals = sorted(y - quadratic_value(coef, x) for x, y in zip(xs, ys))
    quad_q = [base + empirical_quantile(residuals, tau) for tau in LEVELS]
    # --- kernel-local weighted quantiles --------------------------------
    weights = [math.exp(-0.5 * ((x - x_case) / KERNEL_BW) ** 2) for x in xs]
    pairs = sorted(zip(ys, weights))
    local_median = weighted_quantile(pairs, 0.5)
    local_q = [weighted_quantile(pairs, tau) for tau in LEVELS]
    # The quadratic's median is its fitted value re-centred by the residual
    # median (the conditional median, which pinball loss rewards); the
    # local median is the weighted median itself. Average the two sets.
    local_q[2] = local_median
    blended = [(a + b) / 2.0 for a, b in zip(quad_q, local_q)]
    # Keep the blend non-decreasing even under numerical noise.
    for i in range(1, len(blended)):
        if blended[i] < blended[i - 1]:
            blended[i] = blended[i - 1]
    return blended


def fallback_quantiles(p1, hour):
    """House-baseline persistence quantiles, used when comparables are thin."""
    return [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(hour - 1))) for z in Z]


def fallback_forecast(case) -> dict:
    """Persistence forecast for one case, used when the fitted model cannot run.

    Built only from the case's hour-1 print (1 when it is missing or not a
    number) and the hours it asks for (2..24 when the list is missing), so it
    always yields a valid output shape.
    """
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    p1 = max(1, (first or {}).get("viewers") or 1)
    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    return {
        "hours": [
            {"hour": row["hour"], "q": fallback_quantiles(p1, row["hour"])}
            for row in rows
        ]
    }


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first.get("viewers") or 1)
    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]

    series = []
    for comp in case.get("comparables") or []:
        try:
            by_hour = comparable_day_series(comp)
        except (KeyError, ValueError):
            continue
        if by_hour is None:
            continue
        series.append((math.log10(by_hour[1]), by_hour))

    hours = []
    if len(series) >= MIN_COMPS:
        x_all = [x for x, _ in series]
        x_case = min(max(math.log10(p1), min(x_all)), max(x_all))
        for row in rows:
            hour = row["hour"]
            log_q = hour_log_quantiles(series, x_case, hour)
            if log_q is None:
                quantiles = fallback_quantiles(p1, hour)
            else:
                quantiles = [
                    round2(min(max(p1 * math.exp(offset), 0.01), 1_000_000_000.0))
                    for offset in log_q
                ]
                # Safety: enforce non-decreasing after rounding.
                for i in range(1, len(quantiles)):
                    if quantiles[i] < quantiles[i - 1]:
                        quantiles[i] = quantiles[i - 1]
            hours.append({"hour": hour, "q": quantiles})
    else:
        hours = [
            {"hour": row["hour"], "q": fallback_quantiles(p1, row["hour"])}
            for row in rows
        ]
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    paths = sorted(INPUT_DIR.glob("*.in"))
    if not paths:
        raise SystemExit(f"no case files in {INPUT_DIR}")
    for path in paths:
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case, path.name)
        except Exception as exc:  # one bad case must not void the whole run
            print(f"{path.name}: {exc}; writing persistence fallback", file=sys.stderr)
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                case = {}
            forecast = fallback_forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
