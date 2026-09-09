"""
Tests for Task 11: atomic graph wiring on NEXUSRAG_SEMANTIC_PREPROCESSOR flag.

Per B.7 / Q10: create_supervisor_graph() dispatches based on flag.
- Flag=False → _build_legacy_graph() — query_analyzer IS in nodes, semantic_preprocessor NOT
- Flag=True  → _build_new_graph()  — semantic_preprocessor IS in nodes, query_analyzer NOT
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest


class TestAtomicGraphFlag:
    """Verify atomic switch on NEXUSRAG_SEMANTIC_PREPROCESSOR."""

    def test_legacy_graph_has_query_analyzer_not_semantic_preprocessor(self):
        """Legacy graph (flag off): query_analyzer IS in nodes, semantic_preprocessor NOT."""
        from app.services.agents.supervisor import _build_legacy_graph

        graph = _build_legacy_graph()
        node_names = set(graph.nodes.keys())

        assert "query_analyzer" in node_names
        assert "semantic_preprocessor" not in node_names

    def test_new_graph_has_semantic_preprocessor_not_query_analyzer(self):
        """New graph (flag on): semantic_preprocessor IS in nodes, query_analyzer NOT.

        Per Q2: query_analyzer is REMOVED from the new graph.
        """
        from app.services.agents.supervisor import _build_new_graph

        graph = _build_new_graph()
        node_names = set(graph.nodes.keys())

        assert "semantic_preprocessor" in node_names
        assert "query_analyzer" not in node_names

    def test_dispatch_false_returns_legacy_graph(self):
        """Flag=False: create_supervisor_graph() returns legacy-style graph.

        Verifies the dispatch path by checking that when the flag is False,
        the resulting graph has query_analyzer (legacy) and NOT semantic_preprocessor.
        """
        import importlib
        import app.core.config
        import app.services.agents.supervisor as sup_mod

        # Save original
        orig = app.core.config.settings.NEXUSRAG_SEMANTIC_PREPROCESSOR
        try:
            app.core.config.settings.NEXUSRAG_SEMANTIC_PREPROCESSOR = False
            importlib.reload(sup_mod)
            graph = sup_mod.create_supervisor_graph()
            node_names = set(graph.nodes.keys())
            assert "query_analyzer" in node_names
            assert "semantic_preprocessor" not in node_names
        finally:
            app.core.config.settings.NEXUSRAG_SEMANTIC_PREPROCESSOR = orig
            importlib.reload(sup_mod)

    def test_dispatch_true_returns_new_graph(self):
        """Flag=True: create_supervisor_graph() returns new-style graph.

        Verifies the dispatch path by checking that when the flag is True,
        the resulting graph has semantic_preprocessor and NOT query_analyzer.
        """
        import importlib
        import app.core.config
        import app.services.agents.supervisor as sup_mod

        orig = app.core.config.settings.NEXUSRAG_SEMANTIC_PREPROCESSOR
        try:
            app.core.config.settings.NEXUSRAG_SEMANTIC_PREPROCESSOR = True
            importlib.reload(sup_mod)
            graph = sup_mod.create_supervisor_graph()
            node_names = set(graph.nodes.keys())
            assert "semantic_preprocessor" in node_names
            assert "query_analyzer" not in node_names
        finally:
            app.core.config.settings.NEXUSRAG_SEMANTIC_PREPROCESSOR = orig
            importlib.reload(sup_mod)

    def test_new_graph_start_edges_to_semantic_preprocessor(self):
        """START → semantic_preprocessor is the first edge in new graph."""
        from app.services.agents.supervisor import _build_new_graph

        graph = _build_new_graph()
        diagram = graph.get_graph()
        start_edges = [
            (e.source, e.target) for e in diagram.edges
            if e.source == "__start__"
        ]

        assert len(start_edges) == 1
        assert start_edges[0] == ("__start__", "semantic_preprocessor")

    def test_new_graph_semantic_preprocessor_edges_to_supervisor(self):
        """semantic_preprocessor → supervisor is the direct edge after it."""
        from app.services.agents.supervisor import _build_new_graph

        graph = _build_new_graph()
        diagram = graph.get_graph()

        sp_edges = [
            (e.source, e.target) for e in diagram.edges
            if e.source == "semantic_preprocessor"
        ]

        assert ("semantic_preprocessor", "supervisor") in sp_edges

    def test_legacy_graph_has_all_required_nodes(self):
        """Legacy graph has all expected Phase 5 nodes."""
        from app.services.agents.supervisor import _build_legacy_graph

        graph = _build_legacy_graph()
        node_names = set(graph.nodes.keys())

        required = {
            "query_analyzer",
            "supervisor",
            "result_evaluator",
            "rag",
            "write",
            "direct",
            "people",
            "answer_generator",
        }
        for node in required:
            assert node in node_names, f"missing node: {node}"

    def test_new_graph_has_all_required_nodes(self):
        """New graph has all expected Phase 1A nodes."""
        from app.services.agents.supervisor import _build_new_graph

        graph = _build_new_graph()
        node_names = set(graph.nodes.keys())

        required = {
            "semantic_preprocessor",
            "supervisor",
            "result_evaluator",
            "rag",
            "write",
            "direct",
            "people",
            "answer_generator",
        }
        for node in required:
            assert node in node_names, f"missing node: {node}"
