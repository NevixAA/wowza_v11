#!/usr/bin/env bash
# Private price store, LIVE mode (owner 2026-11-24: "so fix it! especially the public odds").
#
#   scripts/price_store_live.sh open [<repo>] [<only-regex>]   before the job's work (this repo's files)
#   scripts/price_store_live.sh push <producer>                after the job's public commit
#   scripts/price_store_live.sh read <repo> <dir> [<only>]     another repo's checkout (e.g. Pro's _v9):
#                                                              fill its missing live files, read-only, no hook
#
# The live price files (config/price_store_paths.json "live") exist ONLY in the private repo
# NevixAA/wowza-exec, branch af-prices-live, at <repo>/main/<path>. `open` copies them into the
# workspace at their usual paths (so the job's code is unchanged and reads identical bytes) and
# installs a pre-commit hook that keeps them out of every public commit; `push` commits what the
# job wrote back to the store, merging with any other run (scripts/price_store_live.py).
#
# Keys: PRICE_STORE_KEY (write deploy key) for writers; a reader job passes only
# PRICE_STORE_RO_KEY. Gate: repo variable PRICE_STORE_LIVE. '1' = live (store only). Anything
# else = ROLLBACK: the job uses the public files as before; where the public tree no longer has a
# live file, `open` fills it read-only from the store so the job never runs on missing data, and
# the job's ordinary public commit re-publishes it. `push` then does nothing. `read` fills the files
# another repo's checkout no longer has (it is never committed).
#
# OPEN / READ FAIL THE JOB when the store is needed but cannot be opened: a writer that ran without
# its history would write a truncated file. Logs carry counts only, never a price.
set -euo pipefail

CMD="${1:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$(command -v python3 || command -v python)"
T="${RUNNER_TEMP:-/tmp}"

_paths() {   # $1 = tag
  KEY="$T/price_store_live$1_key"
  KNOWN="$T/price_store_live$1_known_hosts"
  STORE="$T/price_store_live$1"
  MANIFEST="$T/price_store_live$1_manifest.json"
}

_key() {   # write the deploy key once per tag; export the ssh command when a key exists
  if [ ! -f "$KEY" ]; then
    SECRET="${PRICE_STORE_KEY:-${PRICE_STORE_RO_KEY:-}}"
    if [ -n "$SECRET" ]; then
      ( umask 077; printf '%s\n' "$SECRET" > "$KEY" )
    fi
  fi
  if [ -f "$KEY" ]; then
    # GitHub's published ed25519 host key (docs.github.com, "GitHub's SSH key fingerprints").
    echo "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl" > "$KNOWN"
    export GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$KNOWN"
  fi
}

case "$CMD" in
  open)
    _paths ""
    REPO="${2:-$(basename "${GITHUB_REPOSITORY:-wowza-betting}")}"
    ONLY="${3:-}"
    _key
    if [ ! -f "$KEY" ] && [ "${PRICE_STORE_LIVE:-}" = "1" ]; then
      echo "::error title=price store::PRICE_STORE_LIVE is 1 but no store key (PRICE_STORE_KEY / PRICE_STORE_RO_KEY) reached this job"
      exit 1
    fi
    "$PY" "$HERE/price_store_live.py" open --repo "$REPO" --workspace . --store-dir "$STORE" \
      --manifest "$MANIFEST" --live "${PRICE_STORE_LIVE:-}" ${ONLY:+--only "$ONLY"}
    ;;
  read)
    REPO="${2:?repo}"
    DIR="${3:?checkout dir}"
    ONLY="${4:-}"
    _paths "_$(basename "$DIR")"
    _key
    # PRICE_STORE_OVERWRITE=1 replaces the checkout's own copies too and PRICE_STORE_ANY_BRANCH=1 reads
    # <repo>@main's store files into a checkout of another branch (Pro's data-truth-100 + v9-main captures).
    LIVEFLAG=0
    if [ "${PRICE_STORE_OVERWRITE:-}" = "1" ]; then LIVEFLAG=1; fi
    "$PY" "$HERE/price_store_live.py" open --repo "$REPO" --workspace "$DIR" --store-dir "$STORE" \
      --manifest "$MANIFEST" --live "$LIVEFLAG" --no-hook ${ONLY:+--only "$ONLY"} \
      ${PRICE_STORE_ANY_BRANCH:+--any-branch}
    rm -f "$KEY"
    ;;
  push)
    _paths ""
    PRODUCER="${2:?producer}"
    if [ -f "$KEY" ]; then _key; fi
    rc=0
    "$PY" "$HERE/price_store_live.py" push --producer "$PRODUCER" --workspace . --manifest "$MANIFEST" || rc=$?
    rm -f "$KEY"
    exit "$rc"
    ;;
  *)
    echo "usage: $0 open [repo] [only-regex] | push <producer> | read <repo> <dir> [only-regex]"; exit 2 ;;
esac
