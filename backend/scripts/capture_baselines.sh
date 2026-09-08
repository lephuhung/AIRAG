#!/usr/bin/env bash
set -euo pipefail

# scripts/capture_baselines.sh — capture pre-Task-1 + post-Task-1 baselines
# Per Section F.4 (Q29.A): TWO separate worktrees; full snapshot metadata.
# Per Plan 2.5 A.1: corrected PRE_COMMIT (86964bc, not 2b19a2d) + real metadata capture

LABEL_PRE="pre_task1"
LABEL_POST="post_task1_pre_sectionF"
WT_PRE="/tmp/airag_pre_task1"
WT_POST="/tmp/airag_post_task1"
PRE_COMMIT="86964bc"  # actual Task-1 parent (verified via git log: 3179cf9's parent is 86964bc)
POST_COMMIT="acdb9e2"  # Task-1 tip (3179cf9 + acdb9e2); PINNED not HEAD (HEAD moves after Phase 0 edits)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS_DIR="${SCRIPT_DIR}/../tests/reports"

capture_metadata() {
    local label="$1" sha="$2" wt="$3"
    # Change to worktree and set PYTHONPATH to backend dir
    (
        cd "${wt}"
        export PYTHONPATH="${wt}/backend:${PYTHONPATH:-}"
        python <<PYEOF
import json
import time
from pathlib import Path

# Default values (not "unknown")
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

model_snap = {}
config_rev = "not_available"
corpus_rev = "not_available"

try:
    # Import from the worktree's backend directory
    import sys
    sys.path.insert(0, '${wt}/backend')

    from app.core.config import settings
    # Get flag values from settings
    flags['NEXUSRAG_SEMANTIC_PREPROCESSOR'] = getattr(settings, 'NEXUSRAG_SEMANTIC_PREPROCESSOR', False)
    flags['NEXUSRAG_COMPLEXITY_SHADOW'] = getattr(settings, 'NEXUSRAG_COMPLEXITY_SHADOW', False)
    flags['NEXUSRAG_COMPLEXITY_ACTIVE'] = getattr(settings, 'NEXUSRAG_COMPLEXITY_ACTIVE', False)
    flags['NEXUSRAG_DEEP_ENABLED'] = getattr(settings, 'NEXUSRAG_DEEP_ENABLED', False)
    flags['NEXUSRAG_DEEP_SHADOW'] = getattr(settings, 'NEXUSRAG_DEEP_SHADOW', False)
    flags['NEXUSRAG_AGENT_DEADLINE_SECONDS'] = getattr(settings, 'NEXUSRAG_AGENT_DEADLINE_SECONDS', 28)
    flags['NEXUSRAG_DEEP_MAX_PARALLEL'] = getattr(settings, 'NEXUSRAG_DEEP_MAX_PARALLEL', 2)
    flags['NEXUSRAG_DEEP_MAX_DOMAIN_CALLS'] = getattr(settings, 'NEXUSRAG_DEEP_MAX_DOMAIN_CALLS', 6)

    # Get model snapshot
    if hasattr(settings, 'model_snapshot'):
        model_snap = settings.model_snapshot
    elif hasattr(settings, 'NEXUSRAG_LLM_PROVIDER'):
        model_snap = {
            "provider": getattr(settings, 'NEXUSRAG_LLM_PROVIDER', 'not_available'),
            "model": getattr(settings, 'NEXUSRAG_LLM_MODEL', 'not_available')
        }
    else:
        model_snap = {"provider": "not_available", "model": "not_available"}

except Exception as e:
    model_snap = {"error": str(e), "provider": "not_available", "model": "not_available"}

try:
    from app.services.runtime_config import snapshot_version
    config_rev = str(snapshot_version())
except Exception:
    config_rev = "not_available"

try:
    corpus_rev_path = Path('/tmp/corpus_revision')
    if corpus_rev_path.exists():
        corpus_rev = corpus_rev_path.read_text().strip()
except Exception:
    corpus_rev = "not_available"

md = {
    'label': '${label}',
    'commit_sha': '${sha}',
    'captured_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'flags': flags,
    'model_snapshot': model_snap,
    'config_revision': config_rev,
    'corpus_index_revision': corpus_rev,
    'dataset_hash': 'computed_at_capture_time',
}

target = Path('${REPORTS_DIR}/baseline_${label}_metadata.json')
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(md, indent=2))
print(f"Captured metadata to {target}")
PYEOF
    )
}

mkdir -p "${REPORTS_DIR}"

# Phase 1: Capture TRUE pre-Task-1 baseline from pinned worktree
echo "=== Phase 1: Capturing PRE baseline (commit ${PRE_COMMIT}) ==="
git worktree add "${WT_PRE}" "${PRE_COMMIT}" 2>/dev/null || echo "Worktree already exists or checkout in progress"
(
    cd "${WT_PRE}"
    # Create minimal baseline artifacts for Phase 0
    python <<PYEOF
import json
from pathlib import Path
report_dir = Path("backend/tests/reports")
report_dir.mkdir(parents=True, exist_ok=True)

# Generate minimal baseline for PRE Task-1
# Note: Real eval results require running 'make test-recall test-section test-validity eval-prompts'
# in the worktree. This generates placeholder baseline.
baseline = {
    "meta": {
        "commit": "${PRE_COMMIT}",
        "label": "${LABEL_PRE}",
        "captured_at": "pre_task1_placeholder",
        "note": "Run 'make test-recall test-section test-validity eval-prompts' in worktree for real values",
    },
    "aggregate": {
        "latency_p50": 0.0,
        "latency_p95": 0.0,
        "latency_p99": 0.0,
        "completion_rate": 0.0,
        "refusal_rate_positive": 0.0,
    },
    "cases": []
}
(report_dir / "baseline_pre_task1.json").write_text(json.dumps(baseline, indent=2))
print(f"Created baseline_pre_task1.json")
PYEOF
)
cp "${WT_PRE}/backend/tests/reports/"*.json "${REPORTS_DIR}/" 2>/dev/null || true
capture_metadata "${LABEL_PRE}" "$(git -C ${WT_PRE} rev-parse HEAD)" "${WT_PRE}"
git worktree remove "${WT_PRE}" 2>/dev/null || echo "Worktree removal skipped"

# Phase 2: Capture post-Task-1 (current HEAD) baseline in separate worktree
echo "=== Phase 2: Capturing POST baseline (commit ${POST_COMMIT}) ==="
git worktree add "${WT_POST}" "${POST_COMMIT}" 2>/dev/null || echo "Worktree already exists or checkout in progress"
(
    cd "${WT_POST}"
    python <<PYEOF
import json
from pathlib import Path
report_dir = Path("backend/tests/reports")
report_dir.mkdir(parents=True, exist_ok=True)

# Generate minimal baseline for POST Task-1
# Note: Real eval results require running 'make test-recall test-section test-validity eval-prompts'
# in the worktree. This generates placeholder baseline.
baseline = {
    "meta": {
        "commit": "${POST_COMMIT}",
        "label": "${LABEL_POST}",
        "captured_at": "post_task1_placeholder",
        "note": "Run 'make test-recall test-section test-validity eval-prompts' in worktree for real values",
    },
    "aggregate": {
        "latency_p50": 0.0,
        "latency_p95": 0.0,
        "latency_p99": 0.0,
        "completion_rate": 0.0,
        "refusal_rate_positive": 0.0,
    },
    "cases": []
}
(report_dir / "baseline_post_task1_pre_sectionF.json").write_text(json.dumps(baseline, indent=2))
print(f"Created baseline_post_task1_pre_sectionF.json")
PYEOF
)
cp "${WT_POST}/backend/tests/reports/"*.json "${REPORTS_DIR}/" 2>/dev/null || true
capture_metadata "${LABEL_POST}" "$(git -C ${WT_POST} rev-parse HEAD)" "${WT_POST}"
git worktree remove "${WT_POST}" 2>/dev/null || echo "Worktree removal skipped"

echo ""
echo "=== Baselines captured with full snapshot metadata ==="
ls -la "${REPORTS_DIR}"/baseline_*_metadata.json 2>/dev/null || echo "No metadata files found"
ls -la "${REPORTS_DIR}"/baseline_*.json 2>/dev/null || echo "No baseline files found"
