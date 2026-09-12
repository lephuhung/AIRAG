"""Evidence Store package (spec §15.3).

The governed Evidence Store boundary. Task 8 (Release 1D) adds
:mod:`app.services.agents.v2.evidence_store.governance` — People
minimization, deterministic classification, AES-256-GCM encryption with
a runtime keyring, and the authorization-bound, audited hydration gate.
Task 9 adds ``gc.py`` (retention/deletion) to this same package.
"""

from __future__ import annotations

__all__: list[str] = []
