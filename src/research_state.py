"""
The research clock: does derived evidence advance when its sources do?
=====================================================================
    python -m src.research_state          (writes output/research_state.json + v11_research_health.json)

WHY THIS EXISTS. On 2026-08-25 the raw archives were current to the minute while every movement
summary was two days old:

    output/v11_shadow_snapshots.csv     committed 2026-08-25 09:56   (CI)
    output/v11_movement_summary.csv     committed 2026-08-23 14:01   (by hand)

The movement script had been rewritten to emit six files instead of one, and the workflow's commit
list was never updated — so CI recomputed all six every 30 minutes and threw them away. Nothing
failed. `git add -f` on an unchanged path is a no-op, `|| true` swallowed it, and the run went
green with fresh inputs and frozen conclusions. **Collection health and research health are
different things, and only the first was being measured.**

THE DISTINCTION THIS MODULE EXISTS TO MAKE, which no row count can express:

    NO NEW ELIGIBLE DATA        sources did not advance -> derived files SHOULD be unchanged
    ANALYSIS FAILED TO REFRESH  sources advanced, derived files did not -> STALE

Both look identical from the outside: an unchanged file. The only way to tell them apart is to
compare each derived artifact's `generated_at` against the maximum timestamp of the source it was
built from. That comparison is the whole point.

WHAT IS NOT MEASURED HERE, deliberately. This module does not judge whether the research RESULT
is good. A stale file with a great number in it is still stale, and a fresh file showing no edge
is still healthy. Conflating the two is how a pipeline ends up tuned instead of fixed.

MONOTONICITY. Sample sizes should not shrink. They legitimately can — a methodology change, a
tightened quality rule, a duplicate correction, an explicit quarantine — so a decrease is a WARN
carrying the delta, never an automatic failure. The previous state is read from the committed
`research_state.json`, which is why that file is committed rather than regenerated and discarded.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pandas as pd

import config

CALC_VERSION = "1.0.0"

# Freshness thresholds, set from the workflow's real cadence: v11_collect runs at :25 and :55
# during match hours (08-23 UTC), so ~30 min between runs and an ~9h overnight gap.
#
# WARN at 12h means several consecutive runs failed to refresh while sources moved.
# FAIL at 30h means it has not refreshed across a whole day INCLUDING the overnight window, which
# cannot be explained by the schedule. Neither can fire from one skipped run.
LAG_WARN_H = 12.0
LAG_FAIL_H = 30.0

STATE_FILE = "research_state.json"
HEALTH_FILE = "v11_research_health.json"

# derived artifact -> the sources it is built FROM. A derived file is stale when its
# generated_at predates the newest row in any of its sources by more than the threshold.
DERIVATION = {
    "v11_market_movement_detail.csv": ("v11_shadow_snapshots.csv",),
    "v11_movement_summary.csv": ("v11_shadow_snapshots.csv",),
    "v11_movement_by_residual.csv": ("v11_shadow_snapshots.csv",),
    "v11_movement_by_model.csv": ("v11_shadow_snapshots.csv",),
    "v11_movement_by_time.csv": ("v11_shadow_snapshots.csv",),
    "v11_movement_by_league.csv": ("v11_shadow_snapshots.csv",),
    "v11_residual.csv": ("v11_shadow_log.csv", "v11_graded.csv"),
    "v11_scoreboard.csv": ("v11_graded.csv",),
    "v11_graded.csv": ("v11_shadow_log.csv",),
}

# Where each source's "newest observation" timestamp comes from.
#
# MUST BE A WRITE/OBSERVATION TIMESTAMP, NEVER A FIXTURE DATE. The first version keyed
# v11_graded.csv on `date` — the date the match is PLAYED — and so reported its newest
# observation as 2026-08-31, a week in the FUTURE. Every artifact derived from it then looked
# 133 hours stale when it was in fact committing daily. Fixture dates say what a file is ABOUT;
# only a write timestamp says when it last changed. Same distinction that made an earlier
# freshness check unable to fire at all.
#
# v11_graded.csv has no natural write column, so v11_grade.py now stamps `generated_at`.
SOURCE_TS_COL = {
    "v11_shadow_snapshots.csv": "snapshot_ts",
    "v11_shadow_log.csv": "snapshot_ts",
    "v11_graded.csv": "generated_at",
}

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
_RANK = {PASS: 0, WARN: 1, FAIL: 2}


def _sha() -> str:
    env = os.getenv("GITHUB_SHA")
    if env:
        return env[:12]
    try:
        r = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"],
                           cwd=config.BASE_DIR, capture_output=True, text=True, timeout=15)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _file_mtime(name: str) -> pd.Timestamp | None:
    p = config.OUTPUT_DIR / name
    if not p.exists():
        return None
    return pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC")


def _max_ts(name: str) -> tuple[pd.Timestamp | None, int]:
    """(newest observation timestamp, row count) for a source file."""
    p = config.OUTPUT_DIR / name
    if not p.exists():
        return None, 0
    try:
        d = pd.read_csv(p, low_memory=False)
    except Exception:
        return None, 0
    col = SOURCE_TS_COL.get(name)
    if not col or col not in d.columns:
        return None, len(d)
    t = pd.to_datetime(d[col], errors="coerce", utc=True)
    # A future timestamp is not an observation — it is a fixture date that slipped into a column
    # meant for write times. Discarding them means a mis-keyed column degrades to "no timestamp"
    # (and a visible FAIL for a missing date) rather than to a confident, wrong lag.
    t = t[t <= _now()]
    return (t.max() if t.notna().any() else None), len(d)


def _generated_at(name: str) -> pd.Timestamp | None:
    """A derived file's own generation time.

    Prefers an explicit `generated_at` column, because a file mtime is reset by `git checkout` —
    on a fresh CI runner every file's mtime is checkout time, so mtime alone would report every
    artifact as seconds old and the staleness check could never fire. mtime is the fallback for
    files that predate the column.
    """
    p = config.OUTPUT_DIR / name
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p, nrows=5)
        if "generated_at" in d.columns:
            t = pd.to_datetime(d["generated_at"], errors="coerce", utc=True)
            if t.notna().any():
                return t.max()
    except Exception:
        pass
    return _file_mtime(name)


def _observation_count() -> tuple[int, int]:
    """(distinct observations, fixtures) in the movement detail — DEDUPED.

    Never the raw row count. Pro's archive holds 34,543 rows describing 25,052 distinct
    observations because each ingest is a superset of the last; reporting the row count as a
    sample size inflates n by ~38%, and n is what gates whether a signal graduates.
    """
    p = config.OUTPUT_DIR / "v11_market_movement_detail.csv"
    if not p.exists():
        return 0, 0
    try:
        d = pd.read_csv(p, low_memory=False)
    except Exception:
        return 0, 0
    if "snapshot_id" in d.columns:
        if "ingested_at" in d.columns:
            d = d.sort_values("ingested_at").drop_duplicates("snapshot_id", keep="last")
        else:
            d = d.drop_duplicates("snapshot_id", keep="last")
    fx = d["fixture_id"].nunique() if "fixture_id" in d.columns else 0
    return len(d), int(fx)


def _summary_n() -> dict:
    """Headline sample sizes from the movement summary, as PUBLISHED."""
    p = config.OUTPUT_DIR / "v11_movement_summary.csv"
    out = {"fixture_n": None, "toward_rate": None, "mean_signed_move_pp": None,
           "clv_n": None, "mean_clv_pct": None}
    if not p.exists():
        return out
    try:
        d = pd.read_csv(p)
    except Exception:
        return out
    row = d[d["segment"].astype(str).str.startswith("overall (fixture-level")]
    if row.empty:
        return out
    r = row.iloc[0]
    for k, col in (("fixture_n", "n_fixtures"), ("toward_rate", "toward_rate"),
                   ("mean_signed_move_pp", "mean_signed_move_pp"),
                   ("clv_n", "n_clv"), ("mean_clv_pct", "mean_clv_pct")):
        v = r.get(col)
        out[k] = None if pd.isna(v) else (int(v) if k.endswith("_n") else float(v))
    return out


def _residual_n() -> int | None:
    p = config.OUTPUT_DIR / "v11_residual.csv"
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p)
        r = d[d["scope"] == "overall"]
        return int(r["n"].iloc[0]) if not r.empty else None
    except Exception:
        return None


def _prev_state() -> dict:
    p = config.OUTPUT_DIR / STATE_FILE
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _atomic_write_json(name: str, payload: dict) -> Path:
    """Write, read back, then replace. A half-written JSON parses as invalid, but a truncation
    that happens to land on a closing brace does not — so the read-back is not optional."""
    p = config.OUTPUT_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    json.loads(tmp.read_text(encoding="utf-8"))
    os.replace(tmp, p)
    return p


def _verdict(lag_h: float | None, source_advanced: bool, *,
             derived_dated: bool = True, source_dated: bool = True) -> tuple[str, str]:
    """Freshness verdict for one derived artifact.

    A lag only matters if the SOURCE actually moved. If the source has not advanced, an unchanged
    derived file is correct behaviour — that is the "no new eligible data" case, and calling it
    stale would make the check cry wolf on every quiet night.
    """
    if lag_h is None:
        # Name WHICH side is undateable. The first version said "derived artifact missing or
        # undateable" for both cases, and pointed the blame at v11_scoreboard.csv when the real
        # gap was that its SOURCE (v11_graded.csv) had no generated_at column yet. A monitor that
        # misattributes a fault sends you to the wrong file.
        if not derived_dated and not source_dated:
            return FAIL, "neither this artifact nor its source carries a usable timestamp"
        if not derived_dated:
            return FAIL, "derived artifact missing or has no usable generated_at"
        return WARN, ("cannot judge freshness: the SOURCE has no usable write timestamp yet "
                      "(it gains generated_at on its next CI write); this artifact itself is "
                      "dated and present")
    if not source_advanced:
        return PASS, "source has not advanced; unchanged derived output is correct"
    if lag_h >= LAG_FAIL_H:
        return FAIL, (f"source advanced but derived output is {lag_h:.1f}h behind "
                      f"(>= {LAG_FAIL_H:.0f}h): the analysis is not refreshing")
    if lag_h >= LAG_WARN_H:
        return WARN, f"derived output {lag_h:.1f}h behind its source (>= {LAG_WARN_H:.0f}h)"
    return PASS, f"derived output {lag_h:.1f}h behind its source"


def build() -> tuple[dict, dict]:
    """(research_state, research_health)."""
    now = _now()
    prev = _prev_state()
    prev_derived = (prev.get("derived") or {})

    sources = {}
    for name in sorted(SOURCE_TS_COL):
        ts, rows = _max_ts(name)
        sources[name] = {"rows": rows,
                         "max_observation_ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ") if ts else None,
                         "file_mtime": (_file_mtime(name).strftime("%Y-%m-%dT%H:%M:%SZ")
                                        if _file_mtime(name) else None)}

    prev_sources = (prev.get("source") or {})
    derived, checks = {}, []
    worst = PASS
    for name, srcs in sorted(DERIVATION.items()):
        gen = _generated_at(name)
        # The source timestamp this artifact should have caught up to.
        src_ts = None
        for s in srcs:
            t, _ = _max_ts(s)
            if t is not None and (src_ts is None or t > src_ts):
                src_ts = t
        # Did any source advance since the last state file?
        advanced = False
        for s in srcs:
            cur = sources.get(s, {}).get("max_observation_ts")
            old = (prev_sources.get(s) or {}).get("max_observation_ts")
            if cur and (old is None or cur > old):
                advanced = True
        lag_h = (None if (gen is None or src_ts is None)
                 else round((src_ts - gen).total_seconds() / 3600.0, 2))
        status, why = _verdict(lag_h, advanced, derived_dated=gen is not None,
                               source_dated=src_ts is not None)
        worst = status if _RANK[status] > _RANK[worst] else worst
        derived[name] = {
            "generated_at": gen.strftime("%Y-%m-%dT%H:%M:%SZ") if gen else None,
            "source_max_ts": src_ts.strftime("%Y-%m-%dT%H:%M:%SZ") if src_ts else None,
            "lag_hours": lag_h, "sources": list(srcs),
            "source_advanced_since_last_state": advanced,
            "freshness": status, "why": why,
        }
        checks.append({"artifact": name, "status": status, "lag_hours": lag_h, "why": why})

    obs, fx = _observation_count()
    summ = _summary_n()
    res_n = _residual_n()

    # Monotonicity: n should not shrink. A decrease is WARN with the delta, never an auto-FAIL.
    prev_counts = (prev.get("counts") or {})
    counts, monotonicity = {}, []
    for key, cur in (("movement_observations", obs), ("movement_fixtures", fx),
                     ("movement_summary_fixture_n", summ["fixture_n"]),
                     ("movement_clv_n", summ["clv_n"]), ("residual_n", res_n),
                     ("graded_settled", sources.get("v11_graded.csv", {}).get("rows"))):
        old = prev_counts.get(key)
        delta = None if (cur is None or old is None) else cur - old
        counts[key] = {"current": cur, "previous": old, "delta": delta}
        if delta is not None and delta < 0:
            monotonicity.append({"metric": key, "previous": old, "current": cur, "delta": delta})
            worst = WARN if _RANK[WARN] > _RANK[worst] else worst

    state = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _sha(),
        "calculation_version": CALC_VERSION,
        "source": sources,
        "derived": derived,
        "counts": {k: v["current"] for k, v in counts.items()},
    }

    health = {
        "generated_at": state["generated_at"],
        "git_sha": state["git_sha"],
        "calculation_version": CALC_VERSION,
        "raw": {"snapshots_rows": sources.get("v11_shadow_snapshots.csv", {}).get("rows"),
                "latest_snapshot_ts": sources.get("v11_shadow_snapshots.csv", {})
                .get("max_observation_ts")},
        "grading": {"graded_rows": sources.get("v11_graded.csv", {}).get("rows"),
                    "latest_result_ts": sources.get("v11_graded.csv", {})
                    .get("max_observation_ts")},
        "movement": {
            "n_distinct_observations": obs, "n_fixtures": fx,
            "published_fixture_n": summ["fixture_n"],
            "toward_rate": summ["toward_rate"],
            "mean_signed_move_pp": summ["mean_signed_move_pp"],
            "clv_n": summ["clv_n"], "mean_clv_pct": summ["mean_clv_pct"],
            "derived_generated_at": derived.get("v11_movement_summary.csv", {})
            .get("generated_at"),
            "lag_hours": derived.get("v11_movement_summary.csv", {}).get("lag_hours"),
            "freshness_status": derived.get("v11_movement_summary.csv", {}).get("freshness"),
            "delta_fixtures_since_previous": counts["movement_fixtures"]["delta"],
        },
        "residual": {
            "n_fixtures": res_n,
            "derived_generated_at": derived.get("v11_residual.csv", {}).get("generated_at"),
            "lag_hours": derived.get("v11_residual.csv", {}).get("lag_hours"),
            "freshness_status": derived.get("v11_residual.csv", {}).get("freshness"),
            "delta_n_since_previous": counts["residual_n"]["delta"],
        },
        "monotonicity_warnings": monotonicity,
        "checks": checks,
        "overall": worst,
    }
    return state, health


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Research freshness clock")
    ap.add_argument("--check", action="store_true",
                    help="report only; exit 1 if overall is FAIL")
    a = ap.parse_args()
    state, health = build()

    print(f"[research_state] calc v{CALC_VERSION}  sha {state['git_sha']}")
    print("  sources:")
    for k, v in state["source"].items():
        print(f"    {k:32} {v['rows']:>7,} rows  newest {v['max_observation_ts']}")
    print("  derived:")
    for k, v in state["derived"].items():
        lag = "n/a" if v["lag_hours"] is None else f"{v['lag_hours']:+.1f}h"
        print(f"    [{v['freshness']:4}] {k:38} lag {lag:>8}  {v['why'][:60]}")
    print("  counts:")
    for k, v in health["movement"].items():
        if k.startswith(("n_", "published", "clv_n")):
            print(f"    movement.{k:30} {v}")
    print(f"    residual.n_fixtures{'':13} {health['residual']['n_fixtures']}")
    if health["monotonicity_warnings"]:
        print("  MONOTONICITY WARNINGS (n decreased):")
        for m in health["monotonicity_warnings"]:
            print(f"    {m['metric']}: {m['previous']} -> {m['current']} ({m['delta']:+})")
    print(f"\n[research_state] overall: {health['overall']}")

    if not a.check:
        p1 = _atomic_write_json(STATE_FILE, state)
        p2 = _atomic_write_json(HEALTH_FILE, health)
        print(f"[research_state] wrote {p1.name} and {p2.name}")
    return 1 if (a.check and health["overall"] == FAIL) else 0


if __name__ == "__main__":
    raise SystemExit(main())
