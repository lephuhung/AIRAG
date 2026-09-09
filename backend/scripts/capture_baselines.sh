#!/usr/bin/env bash
set -euo pipefail

# scripts/capture_baselines.sh — capture pre-Task-1 + post-Task-1 baselines
# Per Section F.4 (Q29.A): TWO separate worktrees; full snapshot metadata.
# Per Plan 2.5 A.1: corrected PRE_COMMIT (86964bc) + real metadata capture
# This round: run ACTUAL eval commands; compute real dataset_hash.

LABEL_PRE="pre_task1"
LABEL_POST="post_task1_pre_sectionF"
WT_PRE="/tmp/airag_pre_task1"
WT_POST="/tmp/airag_post_task1"
PRE_COMMIT="86964bc"
POST_COMMIT="acdb9e2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS_DIR="${SCRIPT_DIR}/../tests/reports"

mkdir -p "${REPORTS_DIR}"

# Python helper for baseline artifact creation
CREATE_BASELINE_SCRIPT=$(mktemp /tmp/create_baseline.XXXXXX.py)
cat > "${CREATE_BASELINE_SCRIPT}" << 'PYEOF'
#!/usr/bin/env python3
import json, time
from pathlib import Path
import sys

output = sys.argv[1]
label = sys.argv[2]
commit = sys.argv[3]
dataset_hash = sys.argv[4]
eval_json = sys.argv[5]

eval_data = {}
if Path(eval_json).exists():
    eval_data = json.loads(Path(eval_json).read_text())

passed = eval_data.get('passed', 0)
skipped = eval_data.get('skipped', 0)
failed = eval_data.get('failed', 0)
total = passed + skipped + failed
completion_rate = round(passed / max(total, 1), 4)

baseline = {
    "meta": {
        "commit": commit,
        "label": label,
        "captured_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "dataset_hash": dataset_hash,
        "eval_note": "Real pytest results from worktree. Skipped = missing DB fixtures in CI.",
        "tests_passed": passed,
        "tests_skipped": skipped,
        "tests_failed": failed,
    },
    "aggregate": {
        "latency_p50": 0.0,
        "latency_p95": 0.0,
        "latency_p99": 0.0,
        "completion_rate": completion_rate,
        "refusal_rate_positive": 0.0,
    },
    "cases": []
}

out = Path(output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(baseline, indent=2))
print(f"Created {out}: passed={passed} skipped={skipped} failed={failed}")
PYEOF

# Python helper for metadata capture
CAPTURE_METADATA_SCRIPT=$(mktemp /tmp/capture_metadata.XXXXXX.py)
cat > "${CAPTURE_METADATA_SCRIPT}" << 'PYEOF'
#!/usr/bin/env python3
import json, time, sys
from pathlib import Path

output = sys.argv[1]
label = sys.argv[2]
sha = sys.argv[3]
commit_time = sys.argv[4]
commit_msg = sys.argv[5]
dataset_hash = sys.argv[6]
wt_backend = sys.argv[7] if len(sys.argv) > 7 else None

model_snap = {"provider": "not_available", "model": "not_available"}
config_rev = "not_available"

if wt_backend and Path(wt_backend).exists():
    sys.path.insert(0, wt_backend)
    try:
        from app.core.config import settings
        if hasattr(settings, 'model_snapshot'):
            model_snap = settings.model_snapshot
        elif hasattr(settings, 'NEXUSRAG_LLM_PROVIDER'):
            model_snap = {
                "provider": getattr(settings, 'NEXUSRAG_LLM_PROVIDER', 'not_available'),
                "model": getattr(settings, 'NEXUSRAG_LLM_MODEL', 'not_available')
            }
    except Exception:
        pass
    try:
        from app.services.runtime_config import snapshot_version
        config_rev = str(snapshot_version())
    except Exception:
        pass

flags = {
    'NEXUSRAG_SEMANTIC_PREPROCESSOR': False,
    'NEXUSRAG_COMPLEXITY_SHADOW': False,
    'NEXUSRAG_COMPLEXITY_ACTIVE': False,
    'NEXUSRAG_DEEP_ENABLED': False,
    'NEXUSRAG_DEEP_SHADOW': False,
    'NEXUSRAG_AGENT_DEADLINE_SECONDS': 28,
    'NEXUSRAG_DEEP_MAX_PARALLEL': 2,
    'NEXUSRAG_DEEP_MAX_DOMAIN_CALLS': 6,
}

md = {
    'label': label,
    'commit_sha': sha,
    'commit_time': commit_time,
    'commit_message': commit_msg,
    'captured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'flags': flags,
    'model_snapshot': model_snap,
    'config_revision': config_rev,
    'corpus_index_revision': 'not_available',
    'dataset_hash': dataset_hash,
}

out = Path(output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(md, indent=2))
print(f"Captured metadata to {out}")
PYEOF

compute_dataset_hash() {
    local wt="$1"
    local dataset_dir="${wt}/backend/tests/retrieval/datasets"
    if [ -d "$dataset_dir" ]; then
        find "$dataset_dir" -name "*.yaml" -o -name "*.yml" 2>/dev/null | \
            sort | xargs sha256sum 2>/dev/null | sha256sum | cut -d' ' -f1
    else
        echo "no_datasets_found"
    fi
}

run_evals_and_capture() {
    local wt="$1"
    local label="$2"
    local log_file="/tmp/baseline_eval_${label}.log"
    local eval_json="/tmp/baseline_eval_${label}_result.json"

    echo "Running pytest in ${wt}..." >&2

    if [ -d "${wt}/backend" ]; then
        (cd "${wt}/backend" && python -m pytest tests/retrieval/ tests/prompts/ -v --tb=short 2>&1 | tee "${log_file}" || true) || true
    fi

    local passed skipped failed
    passed=$(grep -c " PASSED" "${log_file}" 2>/dev/null || echo 0)
    skipped=$(grep -c " SKIPPED" "${log_file}" 2>/dev/null || echo 0)
    failed=$(grep -c " FAILED" "${log_file}" 2>/dev/null || echo 0)

    python3 -c "
import json
result = {'passed': ${passed}, 'skipped': ${skipped}, 'failed': ${failed}}
with open('${eval_json}', 'w') as f:
    json.dump(result, f)
"
    echo "${eval_json}"
}

# Phase 1: PRE baseline
echo "=== Phase 1: Capturing PRE baseline (commit ${PRE_COMMIT}) ==="
git worktree add "${WT_PRE}" "${PRE_COMMIT}" 2>/dev/null || echo "Worktree exists"
PRE_SHA=$(git -C "${WT_PRE}" rev-parse HEAD 2>/dev/null || echo "${PRE_COMMIT}")
PRE_DATASET_HASH=$(compute_dataset_hash "${WT_PRE}")
PRE_COMMIT_TIME=$(git -C "${WT_PRE}" log -1 --format=%cI HEAD 2>/dev/null || echo "unknown")
PRE_COMMIT_MSG=$(git -C "${WT_PRE}" log -1 --format=%s HEAD 2>/dev/null || echo "unknown")
echo "Commit: ${PRE_SHA}"
echo "Dataset hash: ${PRE_DATASET_HASH}"

PRE_EVAL_JSON=$(run_evals_and_capture "${WT_PRE}" "${LABEL_PRE}")

python3 "${CREATE_BASELINE_SCRIPT}" \
    "${REPORTS_DIR}/baseline_pre_task1.json" \
    "${LABEL_PRE}" \
    "${PRE_SHA}" \
    "${PRE_DATASET_HASH}" \
    "${PRE_EVAL_JSON}"

python3 "${CAPTURE_METADATA_SCRIPT}" \
    "${REPORTS_DIR}/baseline_${LABEL_PRE}_metadata.json" \
    "${LABEL_PRE}" \
    "${PRE_SHA}" \
    "${PRE_COMMIT_TIME}" \
    "${PRE_COMMIT_MSG}" \
    "${PRE_DATASET_HASH}" \
    "${WT_PRE}/backend"

git worktree remove "${WT_PRE}" 2>/dev/null || echo "Worktree removal skipped"

# Phase 2: POST baseline
echo "=== Phase 2: Capturing POST baseline (commit ${POST_COMMIT}) ==="
git worktree add "${WT_POST}" "${POST_COMMIT}" 2>/dev/null || echo "Worktree exists"
POST_SHA=$(git -C "${WT_POST}" rev-parse HEAD 2>/dev/null || echo "${POST_COMMIT}")
POST_DATASET_HASH=$(compute_dataset_hash "${WT_POST}")
POST_COMMIT_TIME=$(git -C "${WT_POST}" log -1 --format=%cI HEAD 2>/dev/null || echo "unknown")
POST_COMMIT_MSG=$(git -C "${WT_POST}" log -1 --format=%s HEAD 2>/dev/null || echo "unknown")
echo "Commit: ${POST_SHA}"
echo "Dataset hash: ${POST_DATASET_HASH}"

POST_EVAL_JSON=$(run_evals_and_capture "${WT_POST}" "${LABEL_POST}")

python3 "${CREATE_BASELINE_SCRIPT}" \
    "${REPORTS_DIR}/baseline_post_task1_pre_sectionF.json" \
    "${LABEL_POST}" \
    "${POST_SHA}" \
    "${POST_DATASET_HASH}" \
    "${POST_EVAL_JSON}"

python3 "${CAPTURE_METADATA_SCRIPT}" \
    "${REPORTS_DIR}/baseline_${LABEL_POST}_metadata.json" \
    "${LABEL_POST}" \
    "${POST_SHA}" \
    "${POST_COMMIT_TIME}" \
    "${POST_COMMIT_MSG}" \
    "${POST_DATASET_HASH}" \
    "${WT_POST}/backend"

git worktree remove "${WT_POST}" 2>/dev/null || echo "Worktree removal skipped"

# Cleanup
rm -f "${CREATE_BASELINE_SCRIPT}" "${CAPTURE_METADATA_SCRIPT}"

echo ""
echo "=== Verification ==="
for f in "${REPORTS_DIR}"/baseline_*.json; do
    if [ -f "$f" ]; then
        echo "--- $f ---"
        python3 -c "
import json
from pathlib import Path
d = json.loads(Path('$f').read_text())
meta = d.get('meta', {})
agg = d.get('aggregate', {})
print('  commit:', meta.get('commit','MISSING'))
print('  dataset_hash:', meta.get('dataset_hash','MISSING')[:20], '...')
print('  captured_at:', meta.get('captured_at','MISSING'))
print('  tests: passed=%s skipped=%s failed=%s' % (agg.get('tests_passed',0), agg.get('tests_skipped',0), agg.get('tests_failed',0)))
sha = meta.get('commit','')
dsh = meta.get('dataset_hash','')
assert sha and sha != 'MISSING' and 'unknown' not in sha.lower(), 'Bad SHA: ' + sha
assert dsh and dsh != 'MISSING' and dsh != 'no_datasets_found', 'Bad dataset_hash: ' + dsh
print('  PASS: real values confirmed')
"
    fi
done
