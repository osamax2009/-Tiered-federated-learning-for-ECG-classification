#!/usr/bin/env bash
#
# Run the federated comparison and keep a full log of it.
#
#   ./run.sh                       smoke config, quick check that everything works
#   ./run.sh full                  full-scale run, 30 rounds
#   ./run.sh full --rounds 10      any trailing argument is passed to the CLI
#   CONFIG=configs/my.yaml ./run.sh
#
# Logs land in logs/<config>_<timestamp>.log; the machine-readable result is written to
# results/ by the run itself. Both the console and the log get the same output.

set -euo pipefail

cd "$(dirname "$0")"

PROFILE="${1:-smoke}"
[[ $# -gt 0 ]] && shift

CONFIG="${CONFIG:-configs/${PROFILE}.yaml}"
if [[ ! -f "$CONFIG" ]]; then
    echo "no such config: $CONFIG" >&2
    echo "available: $(ls configs/*.yaml | tr '\n' ' ')" >&2
    exit 1
fi

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if [[ -x .venv/bin/python ]]; then
        PYTHON=.venv/bin/python
    else
        PYTHON=python3
    fi
fi

if ! "$PYTHON" -c "import tieredfl" 2>/dev/null; then
    echo "tieredfl is not importable by $PYTHON. Set it up with:" >&2
    echo "  python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
    exit 1
fi

# The config carries its own data_root; DATA_ROOT overrides it when the datasets live
# somewhere else on this machine.
ROOT_ARGS=()
[[ -n "${DATA_ROOT:-}" ]] && ROOT_ARGS=(--data-root "$DATA_ROOT")

mkdir -p logs results
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/$(basename "$CONFIG" .yaml)_${STAMP}.log"

{
    echo "config      $CONFIG"
    echo "data root   ${DATA_ROOT:-from config}"
    echo "started     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "python      $("$PYTHON" --version 2>&1)"
    echo "commit      $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
    echo "packages    $("$PYTHON" -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)')"
    echo
} | tee "$LOG"

# Report what data is present before committing to a long run.
set +e
"$PYTHON" -m tieredfl.cli inspect --config "$CONFIG" "${ROOT_ARGS[@]+"${ROOT_ARGS[@]}"}" 2>&1 | tee -a "$LOG"
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "some datasets hold fewer records than requested; continuing" | tee -a "$LOG"
fi
set -e
echo | tee -a "$LOG"

START=$SECONDS
set +e
"$PYTHON" -m tieredfl.cli run --config "$CONFIG" "${ROOT_ARGS[@]+"${ROOT_ARGS[@]}"}" "$@" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

{
    echo
    echo "exit status $STATUS after $(( (SECONDS - START) / 60 ))m $(( (SECONDS - START) % 60 ))s"
    echo "log         $LOG"
} | tee -a "$LOG"

exit "$STATUS"
