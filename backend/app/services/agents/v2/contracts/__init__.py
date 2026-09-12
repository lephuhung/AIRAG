"""Canonical frozen v2 contracts and pure validators (spec §3–§21).

Owner-focused modules, one authoritative shape per fact:

- ``base``          — strict/frozen ``ContractModel`` + mutable ``RuntimeModel``
- ``request``       — persisted request boundary
- ``conversation``  — discourse context and its persisted snapshot
- ``semantic``      — draft/finalized query meaning and revision requirements
- ``locators``      — structured content coordinate system
- ``binding``       — bound documents, roles, provenance, discovery proposals
- ``routing``       — query analysis and route decision
- ``capability``    — frozen capability descriptor/input/output + runtime context
- ``planning``      — target units, tasks, plans, planner inputs
- ``execution``     — capability request/result facts and task summaries
- ``evidence``      — evidence identity, use, storage policy
- ``evaluation``    — coverage and EvidenceEvaluation
- ``synthesis``     — synthesis hydration and claim/use mapping
- ``clarification`` — clarification request/resolution
- ``response``      — rendered citations and final response
- ``state``         — checkpoint aggregate and request-scoped runtime context
- ``validation``    — pure ``validate_*`` functions
"""
