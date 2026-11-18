#!/usr/bin/env bash
# Private store of v11's prediction ledger (owner licence rule; coordinator decision 2026-11-18).
#
#   scripts/prediction_ledger_store.sh open    clone the store, hydrate output/prediction_ledger{,_settled}/
#   scripts/prediction_ledger_store.sh push    copy this run's ledger back, commit + push
#
# WHY PRIVATE. v11's ledger probability is its O/U 2.5 blend p_blend = w*p_model + (1-w)*p_market, at
# least 55% the de-vigged price consensus; with v9's public p_over25 the market value can be backed out.
# It is price-derived, and this repo is PUBLIC, so the ledger lives ONLY in the private repo
# NevixAA/wowza-exec, branch `prediction-ledger-v11`, cloned outside the workspace ($RUNNER_TEMP). Access
# is the price-store write deploy key on wowza-exec (secret PRICE_STORE_KEY, the key the price-store
# dual-write uses). output/prediction_ledger*/ is git-ignored here and never staged by v11_collect.yml
# (tests/test_prediction_ledger_v11.py fails if it is ever tracked). Nothing here prints a file's
# contents: counts and status only.
#
# OPEN never fails the job. It writes `ok=true` to $GITHUB_OUTPUT only when the store is open and hydrated;
# the collector records ledger rows ONLY then (first-write-wins needs the stored keys - a run that wrote
# without them would offer duplicates of every stored key and copy partial files back). The first run
# creates the orphan branch with its marker. PUSH fails the job when the ledger was requested but the
# store could not be opened or pushed, so a lost run is visible (it runs after the public commit).
set -uo pipefail

REPO="${PREDICTION_LEDGER_STORE_REPO:-git@github.com:NevixAA/wowza-exec.git}"
BRANCH="prediction-ledger-v11"
MARKER=".prediction_ledger_store"
DIR="${RUNNER_TEMP:-/tmp}/prediction_ledger_store"
KEY="${RUNNER_TEMP:-/tmp}/prediction_ledger_key"
KNOWN="${RUNNER_TEMP:-/tmp}/prediction_ledger_known_hosts"
FLAG="${RUNNER_TEMP:-/tmp}/prediction_ledger_open_failed"
OUT="${PREDICTION_LEDGER_OUT:-output}"
PARTS="prediction_ledger prediction_ledger_settled"

_ssh() {
  # GitHub's published ed25519 host key (docs.github.com, "GitHub's SSH key fingerprints").
  echo "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl" > "$KNOWN"
  export GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$KNOWN"
}

_fail_open() {
  echo "::warning title=prediction ledger::$1: v11 records no ledger rows this run"
  rm -rf "$DIR"; rm -f "$KEY"; touch "$FLAG"; exit 0
}

_count() { find "$1" -name '*.csv' 2>/dev/null | wc -l; }

case "${1:-}" in
  open)
    rm -f "$FLAG"
    [ -n "${PRICE_STORE_KEY:-}" ] || _fail_open "PRICE_STORE_KEY is not set"
    ( umask 077; printf '%s\n' "$PRICE_STORE_KEY" > "$KEY" )
    _ssh
    rm -rf "$DIR"
    heads=$(git ls-remote --heads "$REPO" "$BRANCH" 2>/dev/null) || _fail_open "private store unreachable"
    if [ -n "$heads" ]; then
      git clone --quiet --depth 1 --branch "$BRANCH" "$REPO" "$DIR" 2>/dev/null || _fail_open "private store clone failed"
      [ -f "$DIR/$MARKER" ] || _fail_open "store marker missing on $BRANCH"
    else
      # First run: a new orphan branch holding only the marker (pushed with the first rows).
      mkdir -p "$DIR" && cd "$DIR" && git init --quiet && git checkout --quiet --orphan "$BRANCH" \
        && git remote add origin "$REPO" || _fail_open "could not initialise the store branch"
      printf 'v11 prediction ledger (src/prediction_ledger_v11.py) - private: price-derived probabilities.\n' > "$MARKER"
      cd - >/dev/null || exit 0
      echo "[prediction_ledger] store branch $BRANCH does not exist yet; it is created on the first push"
    fi
    # Hydrate: the run appends to the stored partitions (first write wins against every stored key).
    for p in $PARTS; do
      rm -rf "${OUT:?}/$p"
      if [ -d "$DIR/$p" ]; then mkdir -p "$OUT" && cp -a "$DIR/$p" "$OUT/$p"; fi
    done
    echo "ok=true" >> "${GITHUB_OUTPUT:-/dev/null}"
    echo "[prediction_ledger] private store open ($(_count "$DIR/prediction_ledger") ledger file(s))"
    ;;
  push)
    if [ -f "$FLAG" ] || [ ! -d "$DIR/.git" ]; then
      rm -f "$KEY"
      echo "::error title=prediction ledger::the private store was not opened, so this run kept no ledger rows"
      exit 1
    fi
    _ssh
    for p in $PARTS; do
      if [ -d "$OUT/$p" ]; then rm -rf "${DIR:?}/$p" && cp -a "$OUT/$p" "$DIR/$p"; fi
    done
    cd "$DIR" || exit 1
    git config user.name  "github-actions[bot]"
    git config user.email "github-actions[bot]@users.noreply.github.com"
    # one path at a time: `git add a b` stages NOTHING when one of them is absent
    for p in "$MARKER" $PARTS; do
      [ -e "$p" ] && git add -A -- "$p"
    done
    if git diff --cached --quiet 2>/dev/null && git rev-parse --verify --quiet HEAD >/dev/null; then
      echo "[prediction_ledger] nothing new"; rm -f "$KEY"; exit 0
    fi
    FILES=$(git diff --cached --name-only 2>/dev/null | wc -l)
    if ! git commit --quiet -m "prediction-ledger v11: $(date -u '+%Y-%m-%d %H:%M UTC') ($FILES file(s))"; then
      rm -f "$KEY"; echo "::error title=prediction ledger::commit in the private store clone failed"; exit 1
    fi
    ok=0
    for i in 1 2 3 4 5; do
      if git push --quiet origin "HEAD:$BRANCH" 2>/dev/null; then ok=1; break; fi
      # Another run pushed first (the collector's concurrency group makes this rare): rebase onto it.
      git pull --quiet --rebase origin "$BRANCH" 2>/dev/null || true
      echo "[prediction_ledger] push race - retry $i"; sleep $((RANDOM % 5 + 3))
    done
    rm -f "$KEY"
    [ "$ok" = "1" ] || { echo "::error title=prediction ledger::push to the private store failed after 5 tries"; exit 1; }
    echo "[prediction_ledger] pushed $FILES file(s) -> private store ($BRANCH)"
    ;;
  *)
    echo "usage: $0 open | push"; exit 2 ;;
esac
