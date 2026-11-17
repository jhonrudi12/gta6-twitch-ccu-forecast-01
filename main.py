"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts Grand Theft Auto VI's launch-day Twitch concurrent viewers, hour by
hour, from the one number the case gives us at the checkpoint (hour 1's print)
plus the launch-day shapes of 37-42 past big game launches shipped inside the
case itself as `comparables`.

Model (fitted offline on those comparables, then frozen as constants below;
the program re-derives the curves from the case file at run time, so the final
case's extra comparables -- the four practice launches -- are used too):

  * Work in log space: for each past launch, r_h = ln(viewers_h / viewers_1)
    for hours h = 2..24 of its release day.
  * MEDIAN curve: the unweighted median of r_h over the "family" pool -- past
    launches that look like this one, a console-only rolling-midnight launch:
    comparables with no PC in `platforms`, or with "midnight local" in their
    `unlock` note. Their median day runs ~0.7x of hour 1 in the small hours,
    back to ~1x by noon UTC, then up to ~2.3x of hour 1 in the 19:00-22:00 UTC
    evening peak (Europe prime time + US afternoon), easing to ~2x at hour 24.
    The Grand Theft Auto VI category's own pre-release history is NOT a launch
    and is excluded from every pool.
  * SPREAD: the weighted standard deviation of r_h over ALL comparables,
    weights = Gaussian kernel on ln(hour-1 print) with bandwidth 0.7 (launches
    of a similar size behave more alike) x3 for console-only launches. The
    spread is floored at half the family pool's own spread, so it cannot
    collapse when the nearest-size neighbours happen to agree (they are the
    flat simultaneous-unlock PC launches), and scaled by 0.85, the factor that
    scored best in leave-one-out validation.
  * Quantiles: q_tau(h) = p1 * exp(median_h + z_tau * sigma_h), z the standard
    normal quantiles at the five scored levels.

Why not the house baseline (persistence)? It assumes every hour stays at hour
1's count. Real console launches do not: hour 1 is Europe + Oceania only (the
US unlocks at 05:00/08:00 UTC), so the day rises to roughly twice hour 1 by
evening. Leave-one-out validation over the comparables, holding each launch
out in turn and scoring it with the competition's own log-scale pinball loss:

    all 37 launches:      model 0.43  vs  baseline 0.93
    console rolling only: model 0.18  vs  baseline 0.28
    (baseline practice loss stated in the kit: about 0.387)

The program is deterministic (no randomness, no clock, no network) and reads
only the case file it is given, so two runs write byte-identical output.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only. It reads every <id>.in in the input
directory and writes <id>.out per case: {"hours": [{"hour": 2, "q": [q05,
q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each once, five
non-decreasing quantiles each above 0 and at most 1e9, rounded to 2 decimals.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import (none is needed here).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
# House baseline spread, used only as a last-resort fallback.
BASE_SIGMA = 0.15

# --- frozen model constants (see module docstring for how they were chosen) ---
# Gaussian kernel bandwidth on ln(hour-1 print) for the spread weights.
KERNEL_BW = 0.7
# Extra weight for console-only launches in the spread pool.
CONSOLE_BONUS = 3.0
# The spread is scaled by this factor (leave-one-out optimum, 0.7-1.0 tried).
SIGMA_SCALE = 0.85
# The spread never falls below this share of the family pool's own spread.
FAMILY_SIGMA_FLOOR = 0.5
# Absolute floor on the log-spread, so quantiles never collapse onto the median.
SIGMA_MIN = 0.10


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _parse_hour(text):
    """ISO 8601 UTC timestamp -> aware datetime (accepts a trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day_curve(comp: dict) -> dict:
    """Hour (1..24, UTC) -> viewers on the comparable's release day.

    The comparable's `hourly` rows run from 14 days before its release day to
    3 days after it; only the 24 hours of the release day itself are kept.
    """
    try:
        base = _parse_hour(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    curve = {}
    for row in comp.get("hourly") or []:
        try:
            viewers = row["viewers"]
            at = _parse_hour(row["from"])
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None or viewers <= 0:
            continue
        delta = (at - base).total_seconds() / 3600.0
        if 0 <= delta < 24:
            curve[int(delta) + 1] = float(viewers)
    return curve


def is_family(comp: dict) -> bool:
    """Is this past launch in the same family as a console-only rolling-midnight launch?

    Console-only: its `platforms` list names no PC. Rolling midnight: its
    `unlock` note says the consoles unlocked at midnight local time. Either
    signal admits it; the Grand Theft Auto VI category's own pre-release
    history is never a launch and is excluded by the caller.
    """
    platforms = str(comp.get("platforms") or "")
    if "PC" not in platforms:
        return True
    unlock = str(comp.get("unlock") or "").lower()
    return "midnight local" in unlock


def weighted_median(pairs: list) -> float:
    """Weighted median of [(value, weight), ...]; falls back to the plain median."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= 0.5 * total:
            return value
    return pairs[-1][0]


def weighted_std(pairs: list) -> float:
    """Weighted standard deviation of [(value, weight), ...]."""
    total = sum(w for _, w in pairs)
    if total <= 0 or len(pairs) < 2:
        return 0.0
    mean = sum(v * w for v, w in pairs) / total
    var = sum(w * (v - mean) ** 2 for v, w in pairs) / total
    return math.sqrt(max(var, 0.0))


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    hours = [row["hour"] for row in case["forecast"]]

    # --- collect the past launches shipped in the case ---
    # Each entry: (hour -> viewers on its release day, is console-only).
    launches = []
    for comp in case.get("comparables") or []:
        if "Grand Theft Auto VI" in str(comp.get("name") or ""):
            continue  # pre-release category history, not a launch
        curve = release_day_curve(comp)
        if 1 not in curve or len(curve) < 12:
            continue
        platforms = str(comp.get("platforms") or "")
        launches.append((curve, "PC" not in platforms, is_family(comp)))

    log_p1 = math.log(p1)
    rows = []
    for h in hours:
        # Family median of ln(viewers_h / viewers_1); all launches if the
        # family pool has nothing for this hour.
        family = [
            (math.log(curve[h] / curve[1]), 1.0)
            for curve, _console, fam in launches
            if fam and h in curve
        ]
        if not family:
            family = [
                (math.log(curve[h] / curve[1]), 1.0)
                for curve, _console, _fam in launches
                if h in curve
            ]
        if family:
            median = weighted_median(family)
            family_sigma = weighted_std(family)
            spread_pairs = []
            for curve, console, _fam in launches:
                if h not in curve:
                    continue
                weight = math.exp(-0.5 * ((math.log(curve[1]) - log_p1) / KERNEL_BW) ** 2)
                if console:
                    weight *= CONSOLE_BONUS
                spread_pairs.append((math.log(curve[h] / curve[1]), weight))
            sigma = weighted_std(spread_pairs) if spread_pairs else family_sigma
            sigma = max(sigma, FAMILY_SIGMA_FLOOR * family_sigma, SIGMA_MIN) * SIGMA_SCALE
        else:
            # No past launch at all: fall back to the house baseline's shape.
            median, sigma = 0.0, BASE_SIGMA * math.sqrt(max(h - 1, 1))
        quantiles = []
        previous = 0.0
        for z in Z:
            q = p1 * math.exp(median + z * sigma)
            q = min(max(q, 0.01), 1_000_000_000.0)
            q = round2(q)
            if q < previous:  # keep the five quantiles non-decreasing
                q = previous
            quantiles.append(q)
            previous = q
        rows.append({"hour": h, "q": quantiles})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
