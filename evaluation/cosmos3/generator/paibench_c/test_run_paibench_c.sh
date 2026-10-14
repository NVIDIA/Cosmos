#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Regression tests for run_paibench_c.sh generation-shard bookkeeping.
#
# The tests extract the real functions from run_paibench_c.sh (no logic is
# duplicated here) and drive them with a stub python that fakes the
# cosmos-framework inference entrypoint. No GPU, network, or venv is needed.
#
# Run:  bash test_run_paibench_c.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_UNDER_TEST="$HERE/run_paibench_c.sh"
[[ -f "$SCRIPT_UNDER_TEST" ]] || { echo "FAIL: $SCRIPT_UNDER_TEST not found" >&2; exit 1; }

PASS=0
FAIL=0

assert_eq() { # assert_eq <actual> <expected> <message>
    if [[ "$1" == "$2" ]]; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        echo "FAIL: $3"
        echo "  expected: $2"
        echo "  actual:   $1"
    fi
}

assert_contains() { # assert_contains <haystack> <needle> <message>
    if [[ "$1" == *"$2"* ]]; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        echo "FAIL: $3"
        echo "  haystack does not contain: $2"
    fi
}

# Extract one function definition verbatim from the script under test and
# evaluate it in this shell so the tests run the shipped code, not a copy.
load_fn() {
    local _fn="$1"
    local _src
    _src="$(sed -n "/^${_fn}()/,/^}/p" "$SCRIPT_UNDER_TEST")"
    [[ -n "$_src" ]] || { echo "FAIL: function ${_fn}() not found in $SCRIPT_UNDER_TEST" >&2; exit 1; }
    eval "$_src"
}

log() { echo "[test-log] $*"; }
die() { echo "[test-die] $*" >&2; exit 9; }

# Stub venv: a bin dir whose python passes heredoc scripts through to the real
# python3 and fakes the framework inference entrypoint by materializing one
# valid (>50 KB) vision.mp4 per JSONL row.
make_stub_env() {
    local _env="$1"
    mkdir -p "$_env/bin"
    cat > "$_env/bin/python" <<'STUB'
#!/usr/bin/env bash
if [[ "${1:-}" == "-" ]]; then
    exec python3 -
fi
if [[ "${1:-}" == "-m" ]]; then
    stub_shard=""; stub_raw=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -i) stub_shard="$2"; shift 2 ;;
            -o) stub_raw="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    export stub_shard stub_raw
    python3 - <<'FAKE'
import json, os, pathlib
shard, raw = pathlib.Path(os.environ["stub_shard"]), pathlib.Path(os.environ["stub_raw"])
for line in shard.read_text().splitlines():
    if not line.strip():
        continue
    name = json.loads(line)["name"]
    d = raw / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "vision.mp4").write_bytes(b"\x00" * 52000)
FAKE
    exit 0
fi
exec python3 "$@"
STUB
    chmod +x "$_env/bin/python"
}

# Shared fixture: 4 tasks on 4 GPUs through the real build_jsonl_and_run().
run_generation() { # run_generation <output_root> <logfile>
    local _out="$1" _logfile="$2"
    PAIBENCH_C_ROOT="$HERE" \
    PAIBENCH_C_DATASET_ROOT="$PAIBENCH_C_OUTPUT_ROOT_FIXTURE_DATASET" \
    PAIBENCH_C_OUTPUT_ROOT="$_out" \
    PAIBENCH_C_CHECKPOINT="Cosmos3-Nano" \
    PAIBENCH_C_NUM_SAMPLES="4" \
    COSMOS3_UV_ENV="$PAIBENCH_C_OUTPUT_ROOT_FIXTURE_STUB" \
    COSMOS3_REPO="$PAIBENCH_C_OUTPUT_ROOT_FIXTURE_REPO" \
    TRITON_CACHE_DIR="$_out/triton" \
    CUDA_VISIBLE_DEVICES="0,1,2,3" \
        build_jsonl_and_run edge >"$_logfile" 2>&1
}

# ---------------------------------------------------------------------------
# Scenario 1: fresh 4-task / 4-GPU run — every shard must report 1 pending
# task (the wc -l bug counted 0 for a 1-row file without a trailing newline)
# and all four videos must be generated.
# ---------------------------------------------------------------------------
test_fresh_run_counts_single_row_shards() {
    local _tmp; _tmp="$(mktemp -d)"
    make_stub_env "$_tmp/stub-venv"
    mkdir -p "$_tmp/dataset" "$_tmp/repo"
    PAIBENCH_C_OUTPUT_ROOT_FIXTURE_DATASET="$_tmp/dataset"
    PAIBENCH_C_OUTPUT_ROOT_FIXTURE_STUB="$_tmp/stub-venv"
    PAIBENCH_C_OUTPUT_ROOT_FIXTURE_REPO="$_tmp/repo"

    load_fn build_jsonl_and_run

    local _rc=0
    run_generation "$_tmp/out" "$_tmp/run.log" || _rc=$?

    assert_eq "$_rc" "0" "fresh run exits 0"
    assert_eq "$(wc -l < "$_tmp/out/Cosmos3-Nano/edge/input.jsonl" | tr -d ' ')" "4" \
        "input.jsonl has a trailing newline so wc -l counts all 4 rows"
    for gpu in 0 1 2 3; do
        assert_contains "$(cat "$_tmp/run.log")" "GPU $gpu: 1 tasks" \
            "GPU $gpu shard with exactly 1 pending row is counted as 1 task"
    done
    assert_eq "$(find "$_tmp/out/Cosmos3-Nano/edge/videos" -name '*.mp4' | wc -l)" "4" \
        "all 4 shard tasks were generated (none skipped as already done)"
    assert_eq "$(find "$_tmp/out/Cosmos3-Nano/edge" -name 'shard_*.jsonl' | wc -l)" "0" \
        "worker shard files are cleaned up"

    rm -rf "$_tmp"
}

# ---------------------------------------------------------------------------
# Scenario 2: fully-complete resume — every shard reports 0 pending, the
# worker-pids array stays empty, and the wait loop must not trip set -u
# (unbound variable) semantics on bash < 4.4.
# ---------------------------------------------------------------------------
test_resume_with_no_pending_tasks() {
    local _tmp; _tmp="$(mktemp -d)"
    make_stub_env "$_tmp/stub-venv"
    mkdir -p "$_tmp/dataset" "$_tmp/repo"
    PAIBENCH_C_OUTPUT_ROOT_FIXTURE_DATASET="$_tmp/dataset"
    PAIBENCH_C_OUTPUT_ROOT_FIXTURE_STUB="$_tmp/stub-venv"
    PAIBENCH_C_OUTPUT_ROOT_FIXTURE_REPO="$_tmp/repo"

    load_fn build_jsonl_and_run

    run_generation "$_tmp/out" "$_tmp/run1.log"
    local _rc=0
    run_generation "$_tmp/out" "$_tmp/run2.log" || _rc=$?

    assert_eq "$_rc" "0" "fully-complete resume exits 0 under set -euo pipefail"
    local _skipped
    _skipped="$(grep -c 'all tasks done, skipping' "$_tmp/run2.log" || true)"
    assert_eq "$_skipped" "4" "all four GPUs report nothing pending on resume"
    assert_eq "$(find "$_tmp/out/Cosmos3-Nano/edge/videos" -name '*.mp4' | wc -l)" "4" \
        "resume regenerates nothing and keeps the four videos"

    rm -rf "$_tmp"
}

# ---------------------------------------------------------------------------
run_tests() {
    test_fresh_run_counts_single_row_shards
    test_resume_with_no_pending_tasks
    echo
    echo "passed: $PASS  failed: $FAIL"
    [[ "$FAIL" -eq 0 ]]
}

run_tests
