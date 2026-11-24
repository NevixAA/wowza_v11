#!/usr/bin/env python3
"""Private price store, LIVE mode: price files live ONLY in the private store (owner 2026-11-24).

Owner 2026-11-16: move every API-Football and OddsAPI price out of the public repos into the private
NevixAA/wowza-exec, "but make sure it doesn't damage the models". Owner 2026-11-24, on seeing prices
still public: "so fix it! especially the public odds". Dual-write (price_store_sync.py) has run since
11-17 and the zero-difference gate was clean (docs/AF_PRICES_PRIVATE_MIGRATION.md, section 9). This
script is the cut-over: the classes listed under "live" in config/price_store_paths.json stop being
public files and become store-only files at the SAME repo-relative paths.

How a job works with it (scripts/price_store_live.sh wraps the keys):

  open   clone the store branch (partial, sparse: only this repo's live files), copy every live file
         into the workspace at its public path, record each file's git blob id (the merge base) in a
         manifest under $RUNNER_TEMP and install a pre-commit hook. The job's code then reads and
         writes exactly the bytes and paths it always did, so models, settlement and CLV see the
         identical inputs (zero-diff by construction).
  push   after the job: every live file the job created or changed is committed to the store on top
         of the freshest store head. A file another run changed meanwhile is merged (CSV: three-way
         line union; ledgers: the row-level merge of src/ledger.py; JSON: three-way top-level keys).
         A file that cannot be merged is kept twice (ours at its path, theirs under _conflicts/) and
         the job goes red: nothing is ever dropped. Deleting a live file is never propagated.
  guard-staged
         the pre-commit hook body: any live path staged for a PUBLIC commit is unstaged, so no
         workflow, script or worktree commit can publish a live price file again.

Rollback (one repo variable): PRICE_STORE_LIVE != '1'. Then `open` only fills live files that the
public tree no longer has (read-only), installs no hook and `push` does nothing, so every writer
works and commits exactly as before the cut-over (and so re-publishes the files: rollback means
public prices again). Logs carry paths, counts and hashes only - never a price.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import price_store_sync as S  # noqa: E402

ROOT = next((p for p in Path(__file__).resolve().parents if (p / "config" / "price_store_paths.json").is_file()),
            Path(__file__).resolve().parents[1])     # scripts/ (v9) or .github/scripts/ (Pro, v11)
CONFIG = ROOT / "config" / "price_store_paths.json"
MANIFEST_NAME = "price_store_live_manifest.json"
HOOK_TAG = "# price_store_live guard"


class MergeConflict(Exception):
    pass


# ── configuration ────────────────────────────────────────────────────────────────────────────────
def load(path: Path = CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def live_classes(cfg: dict, repo: str) -> list[str]:
    return list((cfg.get("live") or {}).get(repo) or [])


def live_patterns(cfg: dict, repo: str) -> list[re.Pattern]:
    rules = cfg["rules"].get(repo) or {}
    return [re.compile(p) for c in live_classes(cfg, repo) for p in rules.get(c, [])]


def is_live(rx: list[re.Pattern], rel: str) -> bool:
    return any(r.search(rel) for r in rx)


def retired(cfg: dict, repo: str) -> list[str]:
    """Classes whose public copy has been REMOVED from HEAD (the CI guard enforces these)."""
    return list((cfg.get("public_retired") or {}).get(repo) or [])


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _walk_roots(rx: list[re.Pattern]) -> set[str]:
    """Workspace directories that can hold a live file: the literal directory prefix of each rule
    ('' = a rule on repo-root files)."""
    out = set()
    for r in rx:
        pat = r.pattern[1:] if r.pattern.startswith("^") else ""
        lit = re.match(r"(?:[A-Za-z0-9_/-]|\[.\-])*", pat).group(0).replace("\\", "")
        out.add(lit.rsplit("/", 1)[0] if "/" in lit else "")
    return {d for d in out if not any(d != o and o and d.startswith(o + "/") for o in out)}


def workspace_live_files(ws: Path, rx: list[re.Pattern]) -> dict[str, Path]:
    """{rel: path} of every regular file under the workspace that a live rule selects."""
    out: dict[str, Path] = {}
    roots = _walk_roots(rx)
    for top in sorted(roots):
        base = ws / top if top else ws
        if not base.exists():
            continue
        if not top:                                          # rules on repo-root files: no recursion
            for p in base.iterdir():
                if p.is_file() and is_live(rx, p.name):
                    out[p.name] = p
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            for fn in filenames:
                p = Path(dirpath) / fn
                rel = p.relative_to(ws).as_posix()
                if is_live(rx, rel):
                    out[rel] = p
    return out


# ── store access ─────────────────────────────────────────────────────────────────────────────────
def _git(args, cwd=None, data=None, env=None, check=True) -> bytes:
    return S._git(args, cwd=cwd, data=data, env=env, check=check)


def clone_sparse(url: str, branch: str, dest: Path, prefix: str, rx: list[re.Pattern],
                 only: re.Pattern | None = None) -> tuple[str, dict]:
    """Partial, shallow clone; check out only this repo's live files. Returns (head, {rel: oid})."""
    if dest.exists():
        shutil.rmtree(dest)
    _git(["clone", "--quiet", "--filter=blob:none", "--no-checkout", "--depth", "1", "--branch", branch, url, str(dest)])
    head = _git(["rev-parse", "HEAD"], cwd=dest).decode().strip()
    if ".price_store" not in S.ls_tree(dest, head, ".price_store"):
        raise S.StoreError("store marker .price_store missing: refusing an unexpected branch")
    tree = {p[len(prefix):]: v[1] for p, v in S.ls_tree(dest, head, prefix).items()
            if is_live(rx, p[len(prefix):]) and (only is None or only.search(p[len(prefix):]))}
    _git(["config", "core.sparseCheckout", "true"], cwd=dest)
    _git(["config", "core.autocrlf", "false"], cwd=dest)
    patterns = ["/.price_store"] + [f"/{prefix}{rel}" for rel in sorted(tree)]
    info = dest / ".git" / "info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "sparse-checkout").write_text("\n".join(patterns) + "\n", encoding="utf-8")
    _git(["checkout", "--quiet", branch], cwd=dest)          # one batched blob fetch for all live files
    return head, tree


def fetch_head(store: Path, branch: str) -> str:
    _git(["fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin",
          f"+refs/heads/{branch}:refs/remotes/origin/{branch}"], cwd=store)
    return _git(["rev-parse", f"refs/remotes/origin/{branch}"], cwd=store).decode().strip()


def read_blob(store: Path, oid: str) -> bytes:
    return _git(["cat-file", "blob", oid], cwd=store)        # lazy-fetches one blob from the store


def write_blob(store: Path, data: bytes) -> str:
    return _git(["hash-object", "-w", "--no-filters", "--stdin"], cwd=store, data=data).decode().strip()


# ── merges (three-way; base = what this run started from) ───────────────────────────────────────
def _lines(b: bytes) -> list[bytes]:
    return b.splitlines(keepends=True) if b else []


def merge_lines(base: bytes, ours: bytes, theirs: bytes) -> bytes:
    """CSV / line files: re-apply this run's added and removed lines on top of theirs."""
    b, o, t = _lines(base), _lines(ours), _lines(theirs)
    if o and t and o[0].rstrip(b"\r\n") != t[0].rstrip(b"\r\n"):
        raise MergeConflict("header differs")
    removed = Counter(b) - Counter(o)
    added = Counter(o) - Counter(b)
    out = []
    for ln in t:
        if removed[ln] > 0:
            removed[ln] -= 1
            continue
        out.append(ln)
    have = Counter(out)
    for ln in o:
        if added[ln] > 0:
            added[ln] -= 1
            if have[ln] > 0:                                  # another run logged the same line
                have[ln] -= 1
                continue
            if out and not out[-1].endswith(b"\n"):
                out[-1] = out[-1] + b"\n"
            out.append(ln)
    return b"".join(out)


def merge_json(base: bytes, ours: bytes, theirs: bytes) -> bytes:
    try:
        B = json.loads(base) if base else {}
        O, T = json.loads(ours), json.loads(theirs)
    except ValueError as e:
        raise MergeConflict(f"json unreadable ({e.__class__.__name__})") from None
    if not all(isinstance(x, dict) for x in (B, O, T)):
        raise MergeConflict("json is not an object")
    out = dict(T)
    for k, v in O.items():
        if k not in B or B[k] != v:
            out[k] = v
    for k in B:
        if k not in O and k in out and out[k] == B[k]:
            del out[k]
    indent = 1 if b"\n " in ours[:200] else None
    return (json.dumps(out, indent=indent, ensure_ascii=False) + ("\n" if ours.endswith(b"\n") else "")).encode("utf-8")


def merge_rows(rel: str, base: bytes, ours: bytes, theirs: bytes) -> bytes | None:
    """Ledgers with a natural key (v9 src/ledger.SYNC_LEDGERS): the row-level merge the public
    ledger sync already uses. None when this repo has no row key for the file."""
    try:
        sys.path.insert(0, str(ROOT))
        from src.ledger import SYNC_LEDGERS, merge_ledger_rows, _read_csv_text  # type: ignore
    except Exception:                                                   # noqa: BLE001 - not v9
        return None
    if rel not in SYNC_LEDGERS:
        return None
    dec = lambda x: _read_csv_text(x.decode("utf-8")) if x else None    # noqa: E731
    merged, _ = merge_ledger_rows(dec(base), dec(ours), dec(theirs), SYNC_LEDGERS[rel])
    buf = io.StringIO()
    merged.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")


def merge(rel: str, base: bytes, ours: bytes, theirs: bytes) -> bytes:
    rows = merge_rows(rel, base, ours, theirs) if rel.endswith(".csv") else None
    if rows is not None:
        return rows
    if rel.endswith(".json"):
        return merge_json(base, ours, theirs)
    if rel.endswith((".csv", ".jsonl", ".txt")):
        return merge_lines(base, ours, theirs)
    raise MergeConflict("binary file changed on both sides")


# ── commands ─────────────────────────────────────────────────────────────────────────────────────
def _manifest_path(a) -> Path:
    return Path(a.manifest or Path(os.environ.get("RUNNER_TEMP", "/tmp")) / MANIFEST_NAME)


def tracked_live(ws: Path, rx: list[re.Pattern]) -> int:
    p = subprocess.run(["git", "ls-files", "-z"], cwd=ws, capture_output=True)
    if p.returncode != 0:
        return 0
    return sum(1 for f in p.stdout.decode("utf-8", "replace").split("\0") if f and is_live(rx, f))


def install_hook(ws: Path, repo: str, config: Path) -> Path:
    gd = Path(_git(["rev-parse", "--git-common-dir"], cwd=ws).decode().strip())
    if not gd.is_absolute():
        gd = (ws / gd).resolve()
    hooks = gd / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    hook = hooks / "pre-commit"
    script = Path(__file__).resolve()
    body = (f"#!/bin/sh\n{HOOK_TAG}: no live price file may enter a public commit\n"
            f"exec '{Path(sys.executable).as_posix()}' '{script.as_posix()}' guard-staged --repo '{repo}' --config '{Path(config).resolve().as_posix()}'\n")
    hook.write_text(body, encoding="utf-8")
    hook.chmod(0o755)
    return hook


def cmd_open(a) -> int:
    cfg = load(a.config)
    rx = live_patterns(cfg, a.repo)
    only = re.compile(a.only) if a.only else None
    ws = Path(a.workspace).resolve()
    mpath = _manifest_path(a)
    man = {"repo": a.repo, "mode": "off", "files": {}}
    if not rx:
        mpath.write_text(json.dumps(man), encoding="utf-8")
        print(f"[price_store_live] {a.repo}: no live classes configured - nothing to do")
        return 0
    live = a.live == "1"
    here = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ws, capture_output=True, text=True).stdout.strip()
    if here != a.branch and not a.any_branch:
        mpath.write_text(json.dumps(man), encoding="utf-8")
        print(f"[price_store_live] {a.repo}: the workspace is on {here or '?'}, not {a.branch} - live mode applies to "
              f"{a.branch} only; this branch keeps its own files")
        return 0
    public_present = tracked_live(ws, rx)
    if not live and public_present:
        mpath.write_text(json.dumps(man), encoding="utf-8")
        print(f"[price_store_live] {a.repo}: live mode off and the public tree holds {public_present} live file(s) - public files are used (pre-cut-over behaviour)")
        return 0
    store = Path(a.store_dir)
    prefix = S.prefix_of(a.repo, a.branch)
    t0 = time.time()
    head, tree = clone_sparse(a.store_url, a.store_branch, store, prefix, rx, only)
    placed = skipped = 0
    for rel, oid in sorted(tree.items()):
        dst = ws / rel
        if not live and dst.exists():
            skipped += 1                                    # rollback: the public copy stands
            continue
        if dst.exists() and blob_id(dst.read_bytes()) == oid:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(store / prefix / rel, dst)
        placed += 1
    bad = [r for r, o in tree.items() if (ws / r).exists() and live and blob_id((ws / r).read_bytes()) != o]
    if bad:
        raise S.StoreError(f"{len(bad)} live file(s) differ from the store after open")
    man.update(mode="live" if live else "rollback", store_head=head, prefix=prefix, branch=a.store_branch,
               store_dir=str(store), url=a.store_url, files=tree, only=a.only or "",
               opened_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    mpath.write_text(json.dumps(man), encoding="utf-8")
    if live and not a.no_hook:
        install_hook(ws, a.repo, Path(a.config))
    print(f"[price_store_live] {a.repo}: store {head[:9]} open ({'live' if live else 'rollback, read-only fill'}): "
          f"{len(tree)} live file(s), {placed} placed, {skipped} already identical/kept, {time.time() - t0:.0f}s")
    return 0


def cmd_push(a) -> int:
    mpath = _manifest_path(a)
    if not mpath.exists():
        print(f"::error title=price store::push without open (no manifest {mpath.name})")
        return 1
    man = json.loads(mpath.read_text(encoding="utf-8"))
    if man.get("mode") != "live":
        print(f"[price_store_live] {man.get('repo')}: mode {man.get('mode')} - nothing to push (public files are the record)")
        return 0
    cfg = load(a.config)
    rx = live_patterns(cfg, man["repo"])
    only = re.compile(man["only"]) if man.get("only") else None
    ws = Path(a.workspace).resolve()
    store, prefix, branch = Path(man["store_dir"]), man["prefix"], man["branch"]
    base_files: dict = man["files"]
    now = {r: p for r, p in workspace_live_files(ws, rx).items() if not only or only.search(r)}
    changed: dict[str, bytes] = {}
    for rel, p in now.items():
        data = p.read_bytes()
        if blob_id(data) != base_files.get(rel):
            changed[rel] = data
    gone = sorted(r for r in base_files if r not in now)
    if gone:
        print(f"::warning title=price store::{len(gone)} live file(s) were deleted in the workspace; the store keeps them (deletes are never propagated)")
    if not changed:
        print(f"[price_store_live] {man['repo']}: no live file changed - nothing to push ({a.producer})")
        return 0
    stamp = time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime())
    for attempt in range(1, a.attempts + 1):
        try:
            head = fetch_head(store, branch)
            cur = {p[len(prefix):]: v[1] for p, v in S.ls_tree(store, head, prefix).items()}
            entries, merged, conflicts = [], 0, []
            for rel, data in sorted(changed.items()):
                ours = blob_id(data)
                theirs, base = cur.get(rel), base_files.get(rel)
                if theirs is None or theirs == base or theirs == ours:
                    oid = write_blob(store, data)
                else:
                    tb = read_blob(store, theirs)
                    bb = read_blob(store, base) if base else b""
                    try:
                        oid = write_blob(store, merge(rel, bb, data, tb))
                        merged += 1
                    except MergeConflict as e:
                        oid = write_blob(store, data)
                        entries.append(f"100644 {theirs}\t_conflicts/{stamp}/{prefix}{rel}")
                        conflicts.append(f"{rel} ({e})")
                entries.append(f"100644 {oid}\t{prefix}{rel}")
            msg = (f"price-store live: {man['repo']} {a.producer} {stamp} "
                   f"({len(changed)} file(s), {merged} merged, {len(conflicts)} conflict(s))\n")
            c = S._commit(store, head, entries, msg, store.parent / f"{store.name}.index")
            if not c:
                print(f"[price_store_live] {man['repo']}: the store already holds this run's files ({a.producer})")
                return 0
            p = subprocess.run(["git", "push", "--quiet", "origin", f"{c}:refs/heads/{branch}"], cwd=store,
                               capture_output=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
            if p.returncode == 0:
                print(f"[price_store_live] {man['repo']}: pushed {a.producer}: {len(changed)} file(s), "
                      f"{merged} merged with another run, {sum(len(d) for d in changed.values())} bytes -> store {c[:9]}")
                if conflicts:
                    print(f"::error title=price store::{len(conflicts)} file(s) could not be merged; both versions are "
                          f"kept (theirs under _conflicts/{stamp}/): " + ", ".join(conflicts[:10]))
                    return 1
                return 0
            print(f"[price_store_live] push race (attempt {attempt}) - retrying")
        except (S.StoreError, subprocess.CalledProcessError) as e:
            print(f"[price_store_live] attempt {attempt} failed: {str(e)[:300]}")
        time.sleep(min(20, 2 + 3 * attempt))
    print(f"::error title=price store::could not push {len(changed)} live file(s) to the private store after {a.attempts} attempts")
    return 1


def cmd_guard_staged(a) -> int:
    cfg = load(a.config)
    rx = live_patterns(cfg, a.repo)
    if not rx:
        return 0
    p = subprocess.run(["git", "diff", "--cached", "--name-only", "-z", "--no-renames"], capture_output=True)
    staged = [f for f in p.stdout.decode("utf-8", "replace").split("\0") if f]
    bad = [f for f in staged if is_live(rx, f)]
    for i in range(0, len(bad), 200):
        subprocess.run(["git", "reset", "-q", "--", *bad[i:i + 200]], capture_output=True)
    if bad:
        print(f"[price_store_live] pre-commit: {len(bad)} private price file(s) kept out of this public commit")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("open")
    o.add_argument("--repo", required=True)
    o.add_argument("--branch", default="main", help="the public branch whose store prefix this job uses")
    o.add_argument("--workspace", default=".")
    o.add_argument("--store-dir", required=True)
    o.add_argument("--store-url", default=os.environ.get("PRICE_STORE_URL", "git@github.com:NevixAA/wowza-exec.git"))
    o.add_argument("--store-branch", default=os.environ.get("PRICE_STORE_BRANCH", "af-prices-live"))
    o.add_argument("--live", default=os.environ.get("PRICE_STORE_LIVE", ""))
    o.add_argument("--only", default="", help="regex: open only these live files (lighter jobs)")
    o.add_argument("--no-hook", action="store_true")
    o.add_argument("--any-branch", action="store_true", help="read <repo>@--branch store files into a checkout of another branch")
    o.add_argument("--manifest")
    o.add_argument("--config", default=str(CONFIG))
    p = sub.add_parser("push")
    p.add_argument("--producer", required=True)
    p.add_argument("--workspace", default=".")
    p.add_argument("--attempts", type=int, default=6)
    p.add_argument("--manifest")
    p.add_argument("--config", default=str(CONFIG))
    g = sub.add_parser("guard-staged")
    g.add_argument("--repo", required=True)
    g.add_argument("--config", default=str(CONFIG))
    a = ap.parse_args(argv)
    try:
        return {"open": cmd_open, "push": cmd_push, "guard-staged": cmd_guard_staged}[a.cmd](a)
    except S.StoreError as e:
        print(f"::error title=price store::{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
