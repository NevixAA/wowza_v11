#!/usr/bin/env python3
"""Private price store: DUAL-WRITE mirror and count + hash check (owner decision 2026-11-17).

API-Football's terms grant no licence to publish its prices, so every class A (raw API-Football),
class B (prices inside our own ledgers / tips / research frames) and class O (OddsAPI-only raw)
file of the PUBLIC repos is mirrored, byte for byte, into the private repo NevixAA/wowza-exec,
branch ``af-prices-live``, at ``<repo>/<branch>/<path>`` (``/`` in a branch name written ``__``),
the same layout as the immutable snapshot branch ``af-prices-archive``.

  sync   mirror every selected file of each public source into the store (add / change / remove)
  check  compare file counts, git blob ids and (with --sha256) sha256 of every selected file,
         public vs store; exit 1 on any difference

Phase 1 (dual-write): the public writers are not touched. They keep writing their public file
exactly as before; this mirror copies the committed bytes into the store. Readers are unchanged.

Bytes are moved as git blobs: the store fetches the public blob by id straight from the public
repo, so a mirrored file is the identical object (same git blob id = same bytes). Nothing here
reads or prints a file's contents: logs carry paths, counts, sizes and hashes only.

The selection (which paths are class A / B / O) is config/price_store_paths.json, generated from
the sha256-verified archive MANIFEST (docs/AF_PRICES_PRIVATE_MIGRATION.md, branch
chore/af-prices-private). Class C (statistics) and class F (football-data.co.uk) are not selected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "price_store_paths.json"
ZERO = "0" * 40
FETCH_CHUNK = 400
GIT_ID = {
    "GIT_AUTHOR_NAME": "github-actions[bot]",
    "GIT_AUTHOR_EMAIL": "github-actions[bot]@users.noreply.github.com",
    "GIT_COMMITTER_NAME": "github-actions[bot]",
    "GIT_COMMITTER_EMAIL": "github-actions[bot]@users.noreply.github.com",
}


class StoreError(RuntimeError):
    pass


def _git(args, cwd=None, data: bytes | None = None, env: dict | None = None, check=True) -> bytes:
    e = dict(os.environ)
    e.setdefault("GIT_TERMINAL_PROMPT", "0")
    if env:
        e.update(env)
    p = subprocess.run(["git", *args], cwd=cwd, input=data, capture_output=True, env=e)
    if check and p.returncode != 0:
        # stderr of git carries refs / paths / counts, never file contents
        raise StoreError(f"git {' '.join(args[:3])} failed ({p.returncode}): {p.stderr.decode(errors='replace')[-400:]}")
    return p.stdout


def load_config(path: Path = CONFIG) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["_rx"] = {repo: [re.compile(p) for p in r["A_O"] + r["B"]] for repo, r in cfg["rules"].items()}
    return cfg


def selected(cfg: dict, repo: str, path: str, branch: str | None = None) -> bool:
    """A class A / B / O path this mirror copies. Once a class is LIVE (repo variable PRICE_STORE_LIVE=1,
    classes under "live" in the config) its files on the source's main branch are written straight
    to the store by the jobs themselves (scripts/price_store_live.py): the mirror must neither copy
    nor remove them there, so they are not selected for <repo>@main."""
    if branch == "main" and os.environ.get("PRICE_STORE_LIVE") == "1":
        live = [re.compile(p) for c in (cfg.get("live") or {}).get(repo, []) for p in cfg["rules"][repo].get(c, [])]
        if any(r.search(path) for r in live):
            return False
    return any(rx.search(path) for rx in cfg["_rx"].get(repo, []))


def _is_rule_path(cfg: dict, repo: str, path: str) -> bool:
    """Any class rule matches (live or not): used so a live file is never seen as 'extra'."""
    return any(rx.search(path) for rx in cfg["_rx"].get(repo, []))


def prefix_of(repo: str, branch: str) -> str:
    return f"{repo}/{branch.replace('/', '__')}/"


def ls_tree(repo_dir, rev: str, prefix: str = "") -> dict:
    """{path: (mode, blob)} of regular files under prefix at rev (trees only, no blob reads)."""
    args = ["ls-tree", "-r", "-z", "--full-tree", rev]
    if prefix:
        args += ["--", prefix]
    out = {}
    for rec in _git(args, cwd=repo_dir).split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, typ, oid = meta.decode().split()
        if typ == "blob" and mode in ("100644", "100755"):
            out[path.decode("utf-8")] = (mode, oid)
    return out


def clone_public(url: str, branch: str, dest: Path) -> str:
    if dest.exists():
        _git(["fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}"], cwd=dest)
        _git(["update-ref", "HEAD", f"refs/remotes/origin/{branch}"], cwd=dest)
    else:
        _git(["clone", "--quiet", "--filter=blob:none", "--no-checkout", "--depth", "1", "--branch", branch, url, str(dest)])
    return _git(["rev-parse", "HEAD"], cwd=dest).decode().strip()


def open_store(url: str, branch: str, dest: Path) -> str:
    if not dest.exists():
        _git(["clone", "--quiet", "--filter=blob:none", "--no-checkout", "--depth", "1", "--branch", branch, url, str(dest)])
    _git(["fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}"], cwd=dest)
    head = _git(["rev-parse", f"refs/remotes/origin/{branch}"], cwd=dest).decode().strip()
    if ".price_store" not in ls_tree(dest, head, ".price_store"):
        raise StoreError("store marker .price_store missing: refusing to write to an unexpected branch")
    return head


def have_object(repo_dir, oid: str) -> bool:
    p = subprocess.run(["git", "cat-file", "-e", oid], cwd=repo_dir, capture_output=True,
                       env={**os.environ, "GIT_NO_LAZY_FETCH": "1"})
    return p.returncode == 0


def fetch_blobs(repo_dir, url: str, oids) -> None:
    need = sorted({o for o in oids if not have_object(repo_dir, o)})
    for i in range(0, len(need), FETCH_CHUNK):
        _git(["-c", "fetch.writeCommitGraph=false", "fetch", "--quiet", "--no-tags", "--no-write-fetch-head",
              url, *need[i:i + FETCH_CHUNK]], cwd=repo_dir)
    missing = [o for o in need if not have_object(repo_dir, o)]
    if missing:
        raise StoreError(f"{len(missing)} blob(s) could not be fetched from the public source")


def blob_repo(dest: Path) -> Path:
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        _git(["init", "--quiet"], cwd=dest)
    return dest


def blob_size(repo_dir, oid: str) -> int:
    return int(_git(["cat-file", "-s", oid], cwd=repo_dir, env={"GIT_NO_LAZY_FETCH": "1"}).decode().strip())


def record_path(repo: str, branch: str) -> str:
    return f"_sync/{repo}__{branch.replace('/', '__')}.json"


def sync_record(store: Path, rev: str, repo: str, branch: str):
    rp = record_path(repo, branch)
    t = ls_tree(store, rev, rp)
    if rp not in t:
        return None
    # one small counts-only JSON: let the partial clone lazy-fetch it from the store
    return json.loads(_git(["cat-file", "blob", t[rp][1]], cwd=store))


def plan(desired: dict, current: dict) -> tuple[dict, dict, list]:
    add = {p: v for p, v in desired.items() if p not in current}
    chg = {p: v for p, v in desired.items() if p in current and current[p] != v}
    rem = sorted(p for p in current if p not in desired)
    return add, chg, rem


def _commit(store: Path, base: str, entries: list[str], msg: str, idx: Path) -> str:
    env = {**GIT_ID, "GIT_INDEX_FILE": str(idx)}
    if idx.exists():
        idx.unlink()
    _git(["read-tree", base], cwd=store, env=env)
    _git(["update-index", "--index-info"], cwd=store, data=("\n".join(entries) + "\n").encode("utf-8"), env=env)
    tree = _git(["write-tree"], cwd=store, env=env).decode().strip()
    base_tree = _git(["rev-parse", f"{base}^{{tree}}"], cwd=store).decode().strip()
    if tree == base_tree:
        return ""
    return _git(["commit-tree", tree, "-p", base], cwd=store, data=msg.encode("utf-8"), env=GIT_ID).decode().strip()


def sync_source(cfg, store: Path, store_url: str, branch: str, work: Path, repo: str, src_branch: str,
                public_url: str, dry_run: bool, max_push_mb: int, log) -> dict:
    pub = work / "pub" / f"{repo}__{src_branch.replace('/', '__')}"
    pub_head = clone_public(public_url, src_branch, pub)
    desired = {p: v for p, v in ls_tree(pub, pub_head).items() if selected(cfg, repo, p, src_branch)}
    pre = prefix_of(repo, src_branch)
    res = {"source": f"{repo}@{src_branch}", "public_commit": pub_head, "files": len(desired)}
    for attempt in range(1, 6):
        base = open_store(store_url, branch, store)
        current = {p[len(pre):]: v for p, v in ls_tree(store, base, pre).items()
                   if selected(cfg, repo, p[len(pre):], src_branch) or not _is_rule_path(cfg, repo, p[len(pre):])}
        add, chg, rem = plan(desired, current)
        res.update(added=len(add), changed=len(chg), removed=len(rem), store_base=base)
        has_record = record_path(repo, src_branch) in ls_tree(store, base, record_path(repo, src_branch))
        if not (add or chg or rem) and has_record:
            res["store_commit"] = base
            res["status"] = "in-sync"
            return res
        if dry_run:
            res["status"] = "dry-run"
            return res
        todo = {**add, **chg}
        fetch_blobs(store, public_url, [oid for _, oid in todo.values()])
        sizes = {p: blob_size(store, oid) for p, (_, oid) in todo.items()}
        res["bytes_written"] = sum(sizes.values())
        # split oversized catch-ups into several commits / pushes (a single huge pack is what
        # made an earlier push fail with "unpacker error")
        chunks, cur, cur_b = [], [], 0
        for p in sorted(todo):
            if cur and cur_b + sizes[p] > max_push_mb * 1_000_000:
                chunks.append(cur); cur, cur_b = [], 0
            cur.append(p); cur_b += sizes[p]
        chunks.append(cur)
        head, ok = base, True
        for n, part in enumerate(chunks, 1):
            entries = [f"{todo[p][0]} {todo[p][1]}\t{pre}{p}" for p in part]
            last = n == len(chunks)
            if last:
                entries += [f"0 {ZERO}\t{pre}{p}" for p in rem]
                meta = {"repo": repo, "branch": src_branch, "public_commit": pub_head, "files": len(desired),
                        "synced_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                oid = _git(["hash-object", "-w", "--no-filters", "--stdin"], cwd=store,
                           data=(json.dumps(meta, indent=1, sort_keys=True) + "\n").encode()).decode().strip()
                entries.append(f"100644 {oid}\t_sync/{repo}__{src_branch.replace('/', '__')}.json")
            msg = (f"price-store sync: {repo}@{src_branch} {pub_head[:9]} part {n}/{len(chunks)} "
                   f"(+{len(add)} ~{len(chg)} -{len(rem)} file(s), {len(desired)} selected)\n")
            c = _commit(store, head, entries, msg, work / "store.index")
            if not c:
                continue
            p = subprocess.run(["git", "push", "--quiet", "origin", f"{c}:refs/heads/{branch}"], cwd=store,
                               capture_output=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
            if p.returncode != 0:
                ok = False
                log(f"[price_store] push race / reject for {repo}@{src_branch} (attempt {attempt}): retrying")
                break
            head = c
        if not ok:
            time.sleep(min(30, 3 * attempt))
            continue
        # verify: the pushed store tree carries exactly the desired blobs under the prefix
        _git(["fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}"], cwd=store)
        tip = _git(["rev-parse", f"refs/remotes/origin/{branch}"], cwd=store).decode().strip()
        after = {p[len(pre):]: v for p, v in ls_tree(store, tip, pre).items()}
        if after != desired:
            a2, c2, r2 = plan(desired, after)
            raise StoreError(f"post-push verify failed for {repo}@{src_branch}: +{len(a2)} ~{len(c2)} -{len(r2)}")
        res.update(store_commit=head, status="pushed", verified=True)
        return res
    raise StoreError(f"push to the private store failed 5 times for {repo}@{src_branch}")


def cmd_sync(a, cfg, log) -> int:
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    store = work / "store"
    out, rc = [], 0
    for repo, br in sources(a, cfg):
        try:
            r = sync_source(cfg, store, a.store_url, a.store_branch, work, repo, br,
                            a.public_url.format(repo=repo), a.dry_run, a.max_push_mb, log)
        except StoreError as e:
            log(f"::error title=price store::{repo}@{br}: {e}")
            r, rc = {"source": f"{repo}@{br}", "status": "error"}, 1
        out.append(r)
        log("[price_store] " + " ".join(f"{k}={v}" for k, v in r.items() if k not in ("store_base",)))
    if a.report:
        Path(a.report).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return rc


def _blob_sha256(repo_dir, oids) -> dict:
    """sha256 of each blob, streamed via cat-file --batch (contents never leave this function)."""
    oids = list(oids)
    if not oids:
        return {}
    p = subprocess.Popen(["git", "cat-file", "--batch"], cwd=repo_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         env={**os.environ, "GIT_NO_LAZY_FETCH": "1"})
    import threading
    threading.Thread(target=lambda: (p.stdin.write(("\n".join(oids) + "\n").encode()), p.stdin.close()), daemon=True).start()
    out = {}
    for oid in oids:
        hdr = p.stdout.readline().decode().split()
        if len(hdr) < 3 or hdr[1] != "blob":
            raise StoreError(f"blob {oid[:12]} unreadable ({' '.join(hdr[1:2])})")
        n, h = int(hdr[2]), hashlib.sha256()
        while n:
            b = p.stdout.read(min(n, 1 << 20)); h.update(b); n -= len(b)
        p.stdout.read(1)
        out[oid] = h.hexdigest()
    p.wait()
    return out


def cmd_check(a, cfg, log) -> int:
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    store = work / "store"
    base = open_store(a.store_url, a.store_branch, store)
    rc, out = 0, []
    for repo, br in sources(a, cfg):
        url = a.public_url.format(repo=repo)
        pub = work / "pub" / f"{repo}__{br.replace('/', '__')}"
        pub_head = clone_public(url, br, pub)
        pre = prefix_of(repo, br)
        have = {p[len(pre):]: v for p, v in ls_tree(store, base, pre).items()
                if selected(cfg, repo, p[len(pre):], br) or not _is_rule_path(cfg, repo, p[len(pre):])}
        rec = sync_record(store, base, repo, br)
        if rec is None:
            log(f"::error title=price store check::{repo}@{br}: no _sync record in the store (never synced)")
            rc = 1
            out.append({"source": f"{repo}@{br}", "status": "NO_RECORD"})
            continue
        pin = rec["public_commit"]
        # The store mirrors the public source as of the recorded commit; compare against exactly
        # that commit, so rows appended after the last sync (lag) are not counted as differences.
        _git(["fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", pin], cwd=pub)
        desired = {p: v for p, v in ls_tree(pub, pin).items() if selected(cfg, repo, p, br)}
        add, chg, rem = plan(desired, have)
        head_sel = {p: v for p, v in ls_tree(pub, pub_head).items() if selected(cfg, repo, p, br)}
        la, lc, lr = plan(head_sel, have)
        r = {"source": f"{repo}@{br}", "pinned_public_commit": pin, "store_commit": base,
             "public_files": len(desired), "store_files": len(have),
             "missing_in_store": len(add), "blob_differs": len(chg), "extra_in_store": len(rem),
             "lag_files_vs_public_head": len(la) + len(lc) + len(lr), "last_sync_utc": rec.get("synced_utc")}
        if a.sha256:
            # Each side's bytes come from its own repo, into its own plain (non-partial) object
            # store: a partial clone's own remote would re-apply its blob:none filter.
            pub_b, sto_b = blob_repo(work / "blobs_public"), blob_repo(work / "blobs_store")
            p_oids = sorted({oid for _, oid in desired.values()})
            s_oids = sorted({oid for _, oid in have.values()})
            fetch_blobs(pub_b, url, p_oids)
            fetch_blobs(sto_b, a.store_url, s_oids)
            hp, hs = _blob_sha256(pub_b, p_oids), _blob_sha256(sto_b, s_oids)
            both = [p for p in desired if p in have]
            r["sha256_compared"] = len(both)
            r["sha256_mismatch"] = sum(1 for p in both if hp[desired[p][1]] != hs[have[p][1]])
            r["bytes_compared"] = sum(blob_size(pub_b, desired[p][1]) for p in both)
        bad = r["missing_in_store"] + r["blob_differs"] + r["extra_in_store"] + (r["sha256_mismatch"] if a.sha256 else 0)
        r["status"] = "EQUAL" if bad == 0 else "DIFFERENT"
        if bad:
            rc = 1
            log(f"::error title=price store check::{repo}@{br}: {bad} difference(s) public vs store")
        out.append(r)
        log("[price_store check] " + " ".join(f"{k}={v}" for k, v in r.items()))
    if a.report:
        Path(a.report).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return rc


def sources(a, cfg):
    if a.source:
        return [tuple(s.split("@", 1)) for s in a.source]
    return [tuple(s) for s in cfg["sources"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["sync", "check"])
    ap.add_argument("--store-url", default=os.environ.get("PRICE_STORE_URL", "git@github.com:NevixAA/wowza-exec.git"))
    ap.add_argument("--store-branch", default=os.environ.get("PRICE_STORE_BRANCH", "af-prices-live"))
    ap.add_argument("--public-url", default="https://github.com/NevixAA/{repo}.git", help="format string with {repo}")
    ap.add_argument("--work", required=True, help="scratch dir outside the workspace (e.g. $RUNNER_TEMP/price_store)")
    ap.add_argument("--source", action="append", help="repo@branch (repeatable); default: all sources in the config")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--dry-run", action="store_true", help="sync: plan only, write nothing")
    ap.add_argument("--sha256", action="store_true", help="check: also sha256 every file on both sides")
    ap.add_argument("--max-push-mb", type=int, default=300)
    ap.add_argument("--report", help="write the counts-only JSON report here")
    a = ap.parse_args(argv)
    cfg = load_config(Path(a.config))

    def log(s):
        print(s, flush=True)
    try:
        return cmd_sync(a, cfg, log) if a.mode == "sync" else cmd_check(a, cfg, log)
    except StoreError as e:
        log(f"::error title=price store::{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
