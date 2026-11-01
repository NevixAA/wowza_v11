"""
Match results from football-data.co.uk — so grading stops depending on what v9 tipped.
=====================================================================================
WHY THIS EXISTS. `v11_grade.py` derives results from v9's public `bets_ledger.csv`, which holds
only fixtures **v9 actually tipped**. Everything else can never be graded:

    settled 94/336 (28%)      League Two 0/24      Serie B 0/11      Ireland 0/5

That is worse than a coverage gap. The graded subset is CONDITIONED ON v9 having disagreed with
the market enough to fire a tip — and that disagreement is the exact variable the movement
research studies. Any result-based or ROI figure computed from it is drawn from a sample selected
on the independent variable. Prices are unaffected (they are captured for the whole board), which
is why movement and CLV were still reportable while ROI was not.

football-data.co.uk publishes full results for every league v11 shadows, free and unauthenticated.
Grading from actual goals removes the conditioning entirely.

TWO FORMATS, established from v9's proven `src/data_loader.py` rather than guessed:

    standard  https://www.football-data.co.uk/mmz4281/{season}/{code}.csv
              Date (dayfirst), HomeTeam, AwayTeam, FTHG, FTAG
    new       https://www.football-data.co.uk/new/{code}.csv
              Date (dayfirst), Season, Home, Away, HG, AG      (all seasons in one file)

CLUB NAMES DIFFER BETWEEN SOURCES and this is where a results join goes silently wrong. v11's
fixture names come from v9's `predictions.csv` (OddsAPI naming); football-data uses its own.
Root CLAUDE.md invariant 11 records the damage: a naive `startswith(first_word)` mapped
`Real Valladolid CF` onto any club starting "Real" and left 46% of standard fixtures with no form
data. So `resolve` here:

  * is LEAGUE-SCOPED — never matches across competitions;
  * requires uniqueness in BOTH directions, so `Bristol` is refused when both `Bristol City` and
    `Bristol Rovers` are present rather than being silently assigned to whichever sorts first;
  * refuses on ambiguity and returns None. An unmatched fixture stays ungraded, which is a
    visible gap. A wrongly matched fixture is a wrong RESULT, which corrupts the research and
    looks like data.

A NOTE ON LOCAL TESTING. This machine sits behind TLS inspection, so
`www.football-data.co.uk` fails certificate verification here even with certifi; the same
requests succeed from GitHub Actions, which is where v9 has downloaded these files for months.
The parsing, resolution and grading logic is therefore unit-tested against synthetic frames built
to the schema above, and the live fetch is exercised on the first CI run. `fetch_results` returns
an empty frame and prints the reason on any network failure, so a blocked fetch degrades to
"nothing newly graded" rather than to wrong results.
"""
from __future__ import annotations

import re
from io import StringIO
from pathlib import Path

import pandas as pd

# DERIVED from the canonical registry (src/league_registry.py, R2 2026-11-01). This used to be a
# hand copy of v9 config.FOOTBALL_DATA_LEAGUES; it drifted (no Saudi Pro League, no K-League 1, and
# "Austria Bundesliga" as a duplicate KEY instead of an alias). scripts/v11_tests.py now checks the
# registry for equality against v9 on every run.
from src import league_registry as LR  # noqa: E402

STD_LEAGUES = dict(LR.STD_RESULT_CODES)
NEW_LEAGUES = dict(LR.NEW_RESULT_CODES)
AF_LEAGUES = tuple(LR.AF_RESULT_LEAGUES)
AF_HISTORY_REL = "af_history.parquet"

STD_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
NEW_URL = "https://www.football-data.co.uk/new/{code}.csv"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Corporate suffixes and prefixes that carry no identity. Deliberately does NOT include anything
# that distinguishes two real clubs — "City", "Rovers", "United", "II", "B" all stay, because
# dropping them is precisely how Manchester City becomes Manchester United.
_NOISE = {"fc", "cf", "afc", "sc", "ac", "as", "ss", "ssc", "cd", "ud", "sd", "rcd", "club",
          "calcio", "kv", "kaa", "rsc", "sk", "fk", "if", "ff", "bk", "gf", "ik", "aik"}


def season_code(today: pd.Timestamp | None = None) -> str:
    """mmz4281 season code, e.g. '2627'. Derived, never hardcoded.

    Root CLAUDE.md: a hardcoded season froze COLLECT_SEASONS at "2025" and cost six weeks of
    uncollected data while the job reported success daily. Football seasons roll in July.
    """
    t = today or pd.Timestamp.now(tz="UTC")
    y = t.year if t.month >= 7 else t.year - 1
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def _norm(name: str) -> str:
    """Lowercase, strip accents/punctuation, drop identity-free corporate tokens."""
    s = str(name or "").lower().strip()
    s = (s.replace("á", "a").replace("à", "a").replace("ä", "a").replace("â", "a")
          .replace("é", "e").replace("è", "e").replace("ë", "e").replace("ê", "e")
          .replace("í", "i").replace("ï", "i").replace("ó", "o").replace("ö", "o")
          .replace("ô", "o").replace("ú", "u").replace("ü", "u").replace("ç", "c")
          .replace("ş", "s").replace("ğ", "g").replace("ı", "i").replace("ø", "o")
          .replace("å", "a").replace("æ", "ae").replace("ñ", "n"))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [t for t in s.split() if t and t not in _NOISE]
    return " ".join(toks) if toks else s.strip()


def resolve(name: str, candidates: list[str]) -> str | None:
    """Match `name` to one of `candidates`, or None. League-scoped by the caller.

    FIRST v9's canonical resolver (src/team_names.py, a byte-identical copy of v9's, checked in
    scripts/v11_tests.py): aliases incl. result spellings, token pairing, ambiguity refused. Without
    it QPR, Wolves, Saint Etienne, LA Galaxy, D.C. United, Atletico Mineiro, Urawa... never resolved
    and 172 fixtures stayed ungraded (Data Truth audit finding 20). THEN the original tiers below:
    exact, normalised-exact, containment unique IN BOTH DIRECTIONS — "Bristol" is contained in both
    "Bristol City" and "Bristol Rovers", so it is refused rather than assigned to whichever sorts first.
    """
    if not name or not candidates:
        return None
    if name in candidates:
        return name
    from src.team_names import resolve_result as _v9_resolve
    hit = _v9_resolve(name, list(candidates))
    if hit:
        return hit
    n = _norm(name)
    if not n:
        return None
    norm_map: dict[str, list[str]] = {}
    for c in candidates:
        norm_map.setdefault(_norm(c), []).append(c)
    exact = norm_map.get(n)
    if exact and len(exact) == 1:
        return exact[0]
    if exact:
        return None                                  # genuinely ambiguous after normalisation
    # Containment, unique both ways.
    fwd = [c for k, cs in norm_map.items() if n and n in k for c in cs]
    bwd = [c for k, cs in norm_map.items() if k and k in n for c in cs]
    hits = {c for c in fwd + bwd}
    return next(iter(hits)) if len(hits) == 1 else None


def _get(url: str) -> pd.DataFrame:
    import requests
    r = requests.get(url, timeout=25, headers=_HEADERS)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return pd.read_csv(StringIO(r.text), encoding="utf-8-sig", on_bad_lines="skip",
                       low_memory=False)


def parse_standard(raw: pd.DataFrame, league: str) -> pd.DataFrame:
    d = pd.DataFrame()
    d["date"] = pd.to_datetime(raw.get("Date"), dayfirst=True, errors="coerce")
    d["home_team"] = raw.get("HomeTeam")
    d["away_team"] = raw.get("AwayTeam")
    d["home_goals"] = pd.to_numeric(raw.get("FTHG"), errors="coerce")
    d["away_goals"] = pd.to_numeric(raw.get("FTAG"), errors="coerce")
    d["league"] = league
    return d.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])


def parse_new(raw: pd.DataFrame, league: str, *, seasons: tuple[str, ...] | None = None,
              today=None) -> pd.DataFrame:
    """football-data /new/ file -> results. The kept seasons are DERIVED from the league's season
    type and the date (current campaign + the one before), never hardcoded: the old literal
    ("2026", "2027") would have silently dropped every calendar-year league on 2028-01-01.
    Exact label match, not substring ("2026" must not match "2025/2026" by accident)."""
    if seasons is None:
        seasons = LR.nf_result_seasons(league, today)
    if "Season" in raw.columns:
        lab = raw["Season"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        raw = raw[lab.isin(set(seasons))].copy()
    d = pd.DataFrame()
    d["date"] = pd.to_datetime(raw.get("Date"), dayfirst=True, errors="coerce")
    d["home_team"] = raw.get("Home")
    d["away_team"] = raw.get("Away")
    d["home_goals"] = pd.to_numeric(raw.get("HG"), errors="coerce")
    d["away_goals"] = pd.to_numeric(raw.get("AG"), errors="coerce")
    d["league"] = league
    return d.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])


def parse_af_history(af: pd.DataFrame, leagues, *, today=None) -> pd.DataFrame:
    """Results for API-Football-only leagues from v9's committed af_history (FTHG/FTAG), limited
    to the league's current + previous campaign by the same derived rule as parse_new."""
    out = []
    if af is None or af.empty or "league" not in af.columns:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals",
                                     "away_goals", "league"])
    t = pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    for lg in leagues:
        g = af[af["league"].astype(str) == lg]
        if g.empty:
            continue
        d = pd.DataFrame({"date": pd.to_datetime(g["date"], errors="coerce"),
                          "home_team": g["home_team"], "away_team": g["away_team"],
                          "home_goals": pd.to_numeric(g.get("FTHG"), errors="coerce"),
                          "away_goals": pd.to_numeric(g.get("FTAG"), errors="coerce"),
                          "league": lg})
        keep = set(LR.nf_result_seasons(lg, t))
        lab = d["date"].map(lambda x: (LR.season_for(lg, x) or {}).get("fd_label")
                            if pd.notna(x) else None)
        d = d[lab.isin(keep)]
        out.append(d.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"]))
    return (pd.concat(out, ignore_index=True) if out else
            pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals",
                                  "league"]))


def _read_v9_af_history() -> pd.DataFrame:
    """v9's PUBLIC committed af_history.parquet: local checkout (V9_LOCAL) first, raw HTTP second."""
    import os
    from io import BytesIO
    loc = os.getenv("V9_LOCAL", "")
    if loc:
        p = Path(loc) / "output" / AF_HISTORY_REL
        if p.exists():
            return pd.read_parquet(p)
    import requests
    from config import V9_RAW_BASE
    r = requests.get(f"{V9_RAW_BASE}/{AF_HISTORY_REL}", timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    return pd.read_parquet(BytesIO(r.content))


def fetch_results(leagues: list[str] | None = None, *, season: str | None = None,
                  quiet: bool = False) -> pd.DataFrame:
    """Results for the given leagues. Empty frame (never an exception) on network failure.

    League names are resolved through the canonical registry, so an alias ("Austria Bundesliga")
    fetches the canonical league once and the rows carry the canonical name."""
    season = season or season_code()
    if leagues is None:
        want = set(LR.LEAGUES)
    else:
        want = {c for c in (LR.canonical_league(x) for x in leagues) if c}
    frames, failed = [], []
    for league, code in STD_LEAGUES.items():
        if league not in want:
            continue
        try:
            frames.append(parse_standard(_get(STD_URL.format(season=season, code=code)), league))
        except Exception as e:                                        # noqa: BLE001
            failed.append(f"{league}({type(e).__name__})")
    for league, code in NEW_LEAGUES.items():
        if league not in want:
            continue
        try:
            frames.append(parse_new(_get(NEW_URL.format(code=code)), league))
        except Exception as e:                                        # noqa: BLE001
            failed.append(f"{league}({type(e).__name__})")
    af_want = [lg for lg in AF_LEAGUES if lg in want]
    if af_want:
        try:
            frames.append(parse_af_history(_read_v9_af_history(), af_want))
        except Exception as e:                                        # noqa: BLE001
            failed.append(f"v9_af_history({type(e).__name__})")
    out = (pd.concat(frames, ignore_index=True) if frames else
           pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals",
                                 "away_goals", "league"]))
    if not out.empty:
        out["total_goals"] = out["home_goals"] + out["away_goals"]
    if not quiet:
        print(f"[results] {len(out):,} results over {out['league'].nunique() if len(out) else 0} "
              f"league(s), season {season}"
              + (f" | UNAVAILABLE: {', '.join(failed)}" if failed else ""))
    return out


def grade_fixtures(fixtures: pd.DataFrame, results: pd.DataFrame, *,
                   quiet: bool = False) -> pd.DataFrame:
    """Attach total_goals / over25_result to `fixtures` (needs league, match, date).

    `match` is "Home vs Away". Matching is league-scoped and date-windowed by +/-1 day, because a
    fixture listed on one calendar day in one source can be the next day in the other when
    kickoff crosses midnight UTC. The window is deliberately narrow: widening it to catch a
    postponement would start matching the REPLAYED fixture, which is a different result.
    """
    f = fixtures.copy()
    f["total_goals"] = pd.NA
    f["over25_result"] = pd.NA
    f["grade_source"] = pd.NA
    if results is None or results.empty or f.empty:
        if not quiet:
            print("[results] no results available; nothing graded")
        return f

    r = results.copy()
    r["date"] = pd.to_datetime(r["date"], errors="coerce")
    fd = pd.to_datetime(f.get("date"), errors="coerce")
    unmatched: list[str] = []

    for lg, idx in f.groupby(f["league"].astype(str)).groups.items():
        rl = r[r["league"].astype(str) == (LR.canonical_league(lg) or lg)]
        if rl.empty:
            continue
        homes = sorted(rl["home_team"].astype(str).unique())
        aways = sorted(rl["away_team"].astype(str).unique())
        for i in idx:
            m = str(f.at[i, "match"])
            if " vs " not in m:
                continue
            h_raw, a_raw = m.split(" vs ", 1)
            h, a = resolve(h_raw.strip(), homes), resolve(a_raw.strip(), aways)
            if not h or not a:
                unmatched.append(f"{lg}: {m}")
                continue
            cand = rl[(rl["home_team"].astype(str) == h) & (rl["away_team"].astype(str) == a)]
            if cand.empty:
                unmatched.append(f"{lg}: {m}")
                continue
            # STRICT +/-1 day (R2 2026-11-01). This used to fall back to ANY date when nothing
            # was near, so a fixture whose result was not yet published could be graded with an
            # EARLIER meeting of the same two clubs -- a wrong result that looks like data. With
            # the season filter now keeping the previous campaign as well, that fallback would
            # fire more often, so it is removed: no near result -> unmatched (NULL).
            if pd.isna(fd.get(i)):
                unmatched.append(f"{lg}: {m} (no date)")
                continue
            cand = cand[(cand["date"] - fd.get(i)).abs() <= pd.Timedelta(days=1)]
            if cand.empty:
                unmatched.append(f"{lg}: {m} (no result within 1 day)")
                continue
            row = cand.sort_values("date").iloc[-1]
            tg = float(row["home_goals"]) + float(row["away_goals"])
            f.at[i, "total_goals"] = tg
            f.at[i, "over25_result"] = 1 if tg > 2.5 else 0
            f.at[i, "grade_source"] = ("v9_af_history" if (LR.canonical_league(lg) or lg) in AF_LEAGUES
                                       else "football-data.co.uk")

    if not quiet:
        got = int(f["over25_result"].notna().sum())
        print(f"[results] graded {got:,}/{len(f):,} fixtures ({got / max(1, len(f)):.1%})"
              + (f"; {len(unmatched)} unmatched" if unmatched else ""))
        for u in unmatched[:8]:
            print(f"    unmatched: {u}")
        if len(unmatched) > 8:
            print(f"    ... and {len(unmatched) - 8} more")
    return f
