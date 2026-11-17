# GTA VI Twitch CCU Forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/jhonrudi12/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/jhonrudi12/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), after the first — for the make.fun
**gta6-twitch-ccu-forecast-01** bounty. The program reads one case file
(`/input/<id>.in`: hour 1's print plus the past launches shipped in the case)
and writes `/output/<id>.out`: for hours 2–24, five quantiles of that hour's
concurrent viewers at levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95.

## Run it

Python 3.13, standard library only, no network at run time:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## The model

Work in log space. For each past big launch, take the ratio of each release-day
hour to its hour-1 print. The forecast's median curve is the median of those
log-ratios over the "family" pool — past launches shaped like this one, a
console-only rolling-midnight launch (no PC in `platforms`, or "midnight
local" in the unlock note). That family runs ~0.7× of hour 1 in the small
hours, back to ~1× by noon UTC, then up to ~2.3× of hour 1 in the 19:00–22:00
UTC evening peak (Europe prime time + US afternoon), easing to ~2× at hour 24.
The spread is the size-weighted standard deviation of the log-ratios over all
past launches (Gaussian kernel on log hour-1 print, bandwidth 0.7, ×3 for
console-only launches), floored at half the family pool's own spread and
scaled by 0.85. Quantiles: `q = p1 · exp(median + z·sigma)`.

Why not persistence (the house baseline)? Hour 1 is Europe + Oceania only —
the US unlocks at 05:00/08:00 UTC — so a console launch day rises to roughly
twice hour 1 by evening instead of staying flat.

Leave-one-out validation over the comparables, scored with the competition's
own log-scale pinball loss (each launch held out in turn):

| set | this entry | house baseline |
| --- | --- | --- |
| all 37 past launches | 0.79 | 0.93 |
| console rolling-midnight only | 0.18 | 0.28 |

(The kit states the house baseline's practice loss as about 0.387.)

The program is deterministic: no randomness, no clock, no network — two runs
write byte-identical output.
