"""
Knowledge Graph Service
========================

Per-workspace Knowledge Graph using LightRAG with configurable LLM + embeddings.

Two graph storage backends (selected via NEXUSRAG_KG_GRAPH_BACKEND in .env):

  networkx (default)
    → File-based per-workspace storage (NetworkX graph + NanoVectorDB).
    → No extra services required.
    → Each workspace's graph is stored separately under backend/data/lightrag/kb_{id}/.
    → Documents within the same workspace share a connected graph.

  neo4j
    → Shared Neo4j instance; workspace isolation via node label "kb_{workspace_id}".
    → Requires Neo4j 5 running (docker-compose includes nexusrag-neo4j service).
    → All documents across all workspaces share one database; queried per-label.
    → Cross-document entity linking within a workspace works out-of-the-box.

Usage:
    kg = KnowledgeGraphService(workspace_id=1)
    await kg.ingest("markdown text from document...")
    result = await kg.query("What are the key themes?", mode="hybrid")
    await kg.cleanup()
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import settings
from app.services.llm import get_embedding_provider, get_llm_provider
from app.services.llm.types import LLMMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider-based adapters for LightRAG
# ---------------------------------------------------------------------------


async def _kg_embed(texts: list[str]) -> np.ndarray:
    """LightRAG-compatible embedding function using the configured provider."""
    provider = get_embedding_provider()
    return await provider.embed(texts)


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

_extraction_context: ContextVar[bool] = ContextVar("extraction_context", default=False)


class KnowledgeGraphService:
    """
    Per-workspace Knowledge Graph service backed by LightRAG.

    Storage backend is selected by NEXUSRAG_KG_GRAPH_BACKEND:
      - "networkx" (default): file-based per-workspace storage
      - "neo4j": shared Neo4j instance with per-workspace node labels
    """

    def __init__(self, workspace_id: int):
        self.workspace_id = workspace_id
        self.working_dir = str(
            settings.BASE_DIR / "data" / "lightrag" / f"kb_{workspace_id}"
        )
        self._rag = None
        self._initialized = False
        
        from app.services.llm_logger import MinIOLoggerService
        self.llm_logger = MinIOLoggerService()

    async def _kg_llm_complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: Optional[list] = None,
        keyword_extraction: bool = False,
        **kwargs,
    ) -> str:
        """
        LightRAG-compatible LLM function using the configured provider.
        Includes exponential-backoff retry for rate-limit errors (HTTP 429)
        and logs interactions to MinIO via LLMLoggerService.
        """
        provider = get_llm_provider()

        messages: list[LLMMessage] = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        if history_messages:
            for msg in history_messages:
                messages.append(LLMMessage(role=msg.get("role", "user"), content=msg.get("content", "")))
        messages.append(LLMMessage(role="user", content=prompt))

        # Identify which model is currently active
        model_name = getattr(provider, "model", "unknown")
        if model_name == "unknown":
            model_name = getattr(provider, "model_name", "unknown")
        if model_name == "unknown":
            if settings.LLM_PROVIDER == "gemini":
                model_name = settings.LLM_MODEL_FAST
            elif settings.LLM_PROVIDER == "ollama":
                model_name = settings.OLLAMA_MODEL
            elif settings.LLM_PROVIDER == "openai":
                model_name = settings.OPENAI_COMPATIBLE_MODEL

        for attempt in range(4):   # up to 3 retries
            try:
                response = await provider.acomplete(
                    messages, temperature=0.0, max_tokens=4096,
                )
                
                # Log extraction info ONLY if we are in an ingestion/extraction context
                if _extraction_context.get():
                    self.llm_logger.log_llm_call(
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        response=response,
                        model=model_name
                    )
                
                return response
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = "429" in err or "rate" in err or "quota" in err or "resource_exhausted" in err
                if is_rate_limit and attempt < 3:
                    wait = 2 ** attempt   # 1s, 2s, 4s
                    logger.warning(
                        f"[kg_llm] rate-limit hit (attempt {attempt + 1}/4) "
                        f"— retrying in {wait}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    raise
        return ""

    async def _get_rag(self):
        """Lazy-initialize LightRAG instance."""
        if self._rag is not None and self._initialized:
            return self._rag

        from lightrag import LightRAG
        from lightrag.utils import wrap_embedding_func_with_attrs
        from lightrag.kg.shared_storage import initialize_pipeline_status

        os.makedirs(self.working_dir, exist_ok=True)

        # Dynamic embedding dimension from the configured provider
        emb_provider = get_embedding_provider()
        embedding_dim = emb_provider.get_dimension()

        backend = settings.NEXUSRAG_KG_GRAPH_BACKEND.lower()

        if backend == "networkx":
            # Detect dimension mismatch when switching providers
            dim_marker = Path(self.working_dir) / ".embedding_dim"
            if dim_marker.exists():
                prev_dim = int(dim_marker.read_text().strip())
                if prev_dim != embedding_dim:
                    logger.warning(
                        f"Embedding dimension changed ({prev_dim} → {embedding_dim}) "
                        f"for workspace {self.workspace_id}. Clearing KG data for rebuild."
                    )
                    shutil.rmtree(self.working_dir)
                    os.makedirs(self.working_dir, exist_ok=True)
            dim_marker.write_text(str(embedding_dim))

        @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=8192)
        async def embedding_func(texts: list[str]) -> np.ndarray:
            return await _kg_embed(texts)

        if backend == "neo4j":
            graph_storage = "Neo4JStorage"
            # LightRAG's Neo4JStorage reads NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD
            # directly from os.environ (not from pydantic-settings).
            # Ensure they are exported here from the app settings so LightRAG can find them.
            import os as _os
            _os.environ["NEO4J_URI"] = settings.NEO4J_URI
            _os.environ["NEO4J_USERNAME"] = settings.NEO4J_USERNAME
            _os.environ["NEO4J_PASSWORD"] = settings.NEO4J_PASSWORD

            # Set PostgreSQL environment variables for LightRAG (PGKVStorage, PGVectorStorage)
            from urllib.parse import urlparse
            url = urlparse(settings.DATABASE_URL.replace("postgresql+asyncpg", "postgresql"))
            _os.environ.setdefault("POSTGRES_HOST", url.hostname or "localhost")
            _os.environ.setdefault("POSTGRES_PORT", str(url.port or 5432))
            _os.environ.setdefault("POSTGRES_USER", url.username or "postgres")
            _os.environ.setdefault("POSTGRES_PASSWORD", url.password or "")
            _os.environ.setdefault("POSTGRES_DATABASE", url.path.lstrip("/") or "postgres")
            _os.environ.setdefault("POSTGRES_ENABLE_VECTOR", "true")

            # Pass workspace label via the LightRAG `workspace` param so all
            # workspaces share one Neo4j DB but remain isolated by node label.
            extra_kwargs = {"workspace": f"kb_{self.workspace_id}"}
            kv_storage = "PGKVStorage"
            vector_storage = "PGVectorStorage"
            doc_status_storage = "JsonDocStatusStorage"
            # Pass Neo4j addon_params
            extra_kwargs["addon_params"] = {
                "language": settings.NEXUSRAG_KG_LANGUAGE,
                "entity_types": settings.NEXUSRAG_KG_ENTITY_TYPES,
            }
            logger.info(
                f"LightRAG using Neo4j backend for workspace {self.workspace_id} "
                f"(label=kb_{self.workspace_id}, uri={settings.NEO4J_URI}) "
                f"with PostgreSQL KV+Vector storage"
            )
        else:
            graph_storage = "NetworkXStorage"
            kv_storage = "PGKVStorage"
            vector_storage = "PGVectorStorage"
            doc_status_storage = "JsonDocStatusStorage"
            extra_kwargs = {
                "workspace": f"kb_{self.workspace_id}"
            }
            logger.info(
                f"LightRAG using NetworkX (file) backend with PostgreSQL KV+Vector storage for workspace {self.workspace_id}"
            )

        # Build LightRAG kwargs
        rag_kwargs: dict = dict(
            working_dir=self.working_dir,
            llm_model_func=self._kg_llm_complete,
            embedding_func=embedding_func,
            chunk_token_size=settings.NEXUSRAG_KG_CHUNK_TOKEN_SIZE,
            enable_llm_cache=True,
            llm_model_max_async=3,   # max 3 concurrent LLM calls → avoids rate limits
            kv_storage=kv_storage,
            vector_storage=vector_storage,
            graph_storage=graph_storage,
            doc_status_storage=doc_status_storage,
            addon_params={
                "language": settings.NEXUSRAG_KG_LANGUAGE,
                "entity_types": settings.NEXUSRAG_KG_ENTITY_TYPES,
            },
        )

        # Merge any backend-specific kwargs (workspace label, etc.)
        rag_kwargs.update(extra_kwargs)

        self._rag = LightRAG(**rag_kwargs)


        await self._rag.initialize_storages()
        await initialize_pipeline_status()
        self._initialized = True

        logger.info(
            f"LightRAG initialized for workspace {self.workspace_id} "
            f"(embedding_dim={embedding_dim}, backend={backend})"
        )
        return self._rag

    async def ingest(self, markdown_content: str, document_id: Optional[int] = None) -> None:
        """
        Ingest markdown content into the knowledge graph.
        LightRAG extracts entities and relationships automatically.
        If document_id is provided, flush the LLM extraction logs to MinIO.
        """
        rag = await self._get_rag()

        if not markdown_content.strip():
            logger.warning(f"Empty content for workspace {self.workspace_id}, skipping KG ingest")
            return

        token = _extraction_context.set(True)
        try:
            await rag.ainsert(markdown_content)
            logger.info(
                f"KG ingested {len(markdown_content)} chars for workspace {self.workspace_id}"
            )

            # Check if entities were actually extracted
            try:
                all_nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
                if not all_nodes:
                    from app.core.config import settings
                    model = (
                        settings.OLLAMA_MODEL
                        if settings.LLM_PROVIDER.lower() == "ollama"
                        else settings.LLM_MODEL_FAST
                    )
                    logger.warning(
                        f"KG extraction produced 0 entities for workspace {self.workspace_id}. "
                        f"Model '{model}' may not support LightRAG's entity extraction format. "
                        f"Consider using a larger model (e.g. qwen3:14b, gemma3:12b) for KG."
                    )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"KG ingest failed for workspace {self.workspace_id}: {e}")
            raise
        finally:
            _extraction_context.reset(token)
            # Flush buffered logs
            if document_id is not None:
                await self.llm_logger.flush_to_minio(
                    workspace_id=self.workspace_id,
                    document_id=document_id
                )

    async def query(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> str:
        """
        Query the knowledge graph.

        Args:
            question: Natural language question
            mode: Query mode — "naive", "local", "global", "hybrid"
            top_k: Number of results

        Returns:
            LightRAG response text with KG-augmented answer
        """
        from lightrag import QueryParam

        rag = await self._get_rag()

        try:
            result = await asyncio.wait_for(
                rag.aquery(
                    question,
                    param=QueryParam(mode=mode, top_k=top_k),
                ),
                timeout=settings.NEXUSRAG_KG_QUERY_TIMEOUT,
            )
            return result or ""
        except asyncio.TimeoutError:
            logger.warning(
                f"KG query timed out after {settings.NEXUSRAG_KG_QUERY_TIMEOUT}s "
                f"for workspace {self.workspace_id}"
            )
            return ""
        except Exception as e:
            logger.error(f"KG query failed for workspace {self.workspace_id}: {e}")
            return ""

    async def cleanup(self) -> None:
        """Finalize storages on shutdown."""
        if self._rag:
            try:
                await self._rag.finalize_storages()
                logger.info(f"KG storages finalized for workspace {self.workspace_id}")
            except Exception as e:
                logger.warning(f"KG cleanup failed for workspace {self.workspace_id}: {e}")
            self._rag = None
            self._initialized = False

    async def delete_project_data(self) -> None:
        """
        Delete all KG data for this knowledge base.

        - NetworkX backend: removes the working directory tree.
        - Neo4j backend: drops all nodes/edges with label kb_{workspace_id}
          in Neo4j, then removes the working directory (KV/vector files).
        """
        backend = settings.NEXUSRAG_KG_GRAPH_BACKEND.lower()

        if backend == "neo4j":
            try:
                from neo4j import AsyncGraphDatabase
                async with AsyncGraphDatabase.driver(
                    settings.NEO4J_URI,
                    auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
                ) as driver:
                    label = f"kb_{self.workspace_id}"
                    async with driver.session() as session:
                        # Delete all nodes with this workspace label (and their edges)
                        await session.run(
                            f"MATCH (n:`{label}`) DETACH DELETE n"
                        )
                    logger.info(
                        f"Deleted Neo4j nodes for workspace {self.workspace_id} "
                        f"(label={label})"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to delete Neo4j data for workspace {self.workspace_id}: {e}"
                )

        # Always remove file-based KV / vector data (working_dir)
        path = Path(self.working_dir)
        if path.exists():
            shutil.rmtree(path)
            logger.info(f"Deleted KG working dir for workspace {self.workspace_id}")

        self._rag = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Knowledge Graph exploration (Phase 9)
    # ------------------------------------------------------------------

    async def get_entities(
        self,
        search: str | None = None,
        entity_type: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict]:
        """
        List all entities in the knowledge graph.

        Returns list of dicts with: name, entity_type, description, degree.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            all_nodes = await storage.get_all_nodes()
        except Exception as e:
            logger.error(f"Failed to get KG nodes for workspace {self.workspace_id}: {e}")
            return []

        entities = []
        for node in all_nodes:
            node_id = node.get("id", "")
            etype = node.get("entity_type", "Unknown")
            desc = node.get("description", "")

            # Filters
            if entity_type and etype.lower() != entity_type.lower():
                continue
            if search and search.lower() not in node_id.lower():
                continue

            # Get degree (number of relationships)
            try:
                degree = await storage.node_degree(node_id)
            except Exception:
                degree = 0

            entities.append({
                "name": node_id,
                "entity_type": etype,
                "description": desc,
                "degree": degree,
            })

        # Sort by degree descending
        entities.sort(key=lambda e: e["degree"], reverse=True)

        return entities[offset:offset + limit]

    async def get_relationships(
        self,
        entity_name: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """
        List relationships in the knowledge graph.

        If entity_name is provided, returns only relationships involving that entity.
        Returns list of dicts with: source, target, description, keywords, weight.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            all_edges = await storage.get_all_edges()
        except Exception as e:
            logger.error(f"Failed to get KG edges for workspace {self.workspace_id}: {e}")
            return []

        relationships = []
        for edge in all_edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")

            if entity_name:
                if entity_name.lower() not in (src.lower(), tgt.lower()):
                    continue

            relationships.append({
                "source": src,
                "target": tgt,
                "description": edge.get("description", ""),
                "keywords": edge.get("keywords", ""),
                "weight": float(edge.get("weight", 1.0)),
            })

        return relationships[:limit]

    async def get_graph_data(
        self,
        center_entity: str | None = None,
        max_depth: int = 3,
        max_nodes: int = 150,
    ) -> dict:
        """
        Export graph data for frontend visualization.

        Returns {nodes: [...], edges: [...], is_truncated: bool}.

        For the Neo4j backend we use a direct Cypher query to bypass the
        LightRAG internal get_knowledge_graph() method, which can fail with
        a 'NoneType has no attribute session' error if LightRAG's driver has
        not been initialised yet for that session.

        For NetworkX backend we fall back to the LightRAG storage API.
        """
        backend = settings.NEXUSRAG_KG_GRAPH_BACKEND.lower()

        if backend == "neo4j":
            return await self._get_graph_data_neo4j(
                center_entity=center_entity,
                max_nodes=max_nodes,
            )

        # ── NetworkX path ──────────────────────────────────────────────────
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            all_nodes = await storage.get_all_nodes()
            all_edges = await storage.get_all_edges()
        except Exception as e:
            logger.error(f"Failed to get KG data for workspace {self.workspace_id}: {e}")
            return {"nodes": [], "edges": [], "is_truncated": False}

        nodes_out = []
        for n in (all_nodes or []):
            nid = n.get("id", "")
            try:
                degree = await storage.node_degree(nid)
            except Exception:
                degree = 0
            nodes_out.append({
                "id": nid,
                "label": nid,
                "entity_type": n.get("entity_type", "Unknown"),
                "degree": degree,
            })
            if len(nodes_out) >= max_nodes:
                break

        edges_out = []
        for e in (all_edges or []):
            edges_out.append({
                "source": e.get("source", ""),
                "target": e.get("target", ""),
                "label": str(e.get("description", ""))[:80],
                "weight": float(e.get("weight", 1.0)),
            })

        return {
            "nodes": nodes_out,
            "edges": edges_out,
            "is_truncated": len(nodes_out) >= max_nodes,
        }

    async def _get_graph_data_neo4j(
        self,
        center_entity: str | None,
        max_nodes: int,
    ) -> dict:
        """Direct Cypher-based graph data export for Neo4j backend.

        Bypasses LightRAG's get_knowledge_graph() to avoid the
        'NoneType has no attribute session' error.
        """
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError:
            logger.warning("neo4j driver not installed")
            return {"nodes": [], "edges": [], "is_truncated": False}

        label = f"kb_{self.workspace_id}"

        try:
            async with AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
            ) as driver:
                async with driver.session(database="neo4j") as session:
                    if center_entity:
                        # Subgraph rooted at center_entity
                        cypher_nodes = f"""
                        MATCH (n:`{label}`)
                        WHERE toLower(n.entity_id) CONTAINS toLower($center)
                        WITH n LIMIT {max_nodes}
                        OPTIONAL MATCH (n)-[r]-(m:`{label}`)
                        RETURN
                            n.entity_id AS entity_name,
                            n.entity_type AS entity_type,
                            m.entity_id AS neighbor,
                            m.entity_type AS neighbor_type,
                            r.description AS rel_desc,
                            startNode(r).entity_id AS rel_src,
                            endNode(r).entity_id AS rel_tgt
                        LIMIT {max_nodes * 3}
                        """
                        result = await session.run(cypher_nodes, center=center_entity)
                    else:
                        # Full workspace overview
                        cypher_nodes = f"""
                        MATCH (n:`{label}`)
                        WITH n LIMIT {max_nodes}
                        OPTIONAL MATCH (n)-[r]-(m:`{label}`)
                        RETURN
                            n.entity_id AS entity_name,
                            n.entity_type AS entity_type,
                            m.entity_id AS neighbor,
                            m.entity_type AS neighbor_type,
                            r.description AS rel_desc,
                            startNode(r).entity_id AS rel_src,
                            endNode(r).entity_id AS rel_tgt
                        LIMIT {max_nodes * 3}
                        """
                        result = await session.run(cypher_nodes)

                    records = await result.data()

            # ── Deduplicate and build output --
            seen_nodes: dict[str, str] = {}  # name → type
            seen_edges: set[tuple] = set()
            edges_out: list[dict] = []

            for rec in records:
                if rec.get("entity_name"):
                    seen_nodes.setdefault(rec["entity_name"], rec.get("entity_type", "Unknown"))
                if rec.get("neighbor"):
                    seen_nodes.setdefault(rec["neighbor"], rec.get("neighbor_type", "Unknown"))
                src = rec.get("rel_src")
                tgt = rec.get("rel_tgt")
                if src and tgt:
                    key = (src, tgt)
                    if key not in seen_edges:
                        seen_edges.add(key)
                        edges_out.append({
                            "source": src,
                            "target": tgt,
                            "label": str(rec.get("rel_desc") or "")[:80],
                            "weight": 1.0,
                        })

            nodes_out = [
                {"id": name, "label": name, "entity_type": etype, "degree": 0}
                for name, etype in seen_nodes.items()
            ]
            # Fill degree from edge counts
            degree_map: dict[str, int] = {}
            for e in edges_out:
                degree_map[e["source"]] = degree_map.get(e["source"], 0) + 1
                degree_map[e["target"]] = degree_map.get(e["target"], 0) + 1
            for n in nodes_out:
                n["degree"] = degree_map.get(n["id"], 0)

            logger.info(
                f"KG graph (Neo4j Cypher): {len(nodes_out)} nodes, "
                f"{len(edges_out)} edges for workspace {self.workspace_id}"
            )
            return {
                "nodes": nodes_out,
                "edges": edges_out,
                "is_truncated": len(nodes_out) >= max_nodes,
            }

        except Exception as e:
            logger.error(f"Failed to fetch graph data from Neo4j for workspace {self.workspace_id}: {e}")
            return {"nodes": [], "edges": [], "is_truncated": False}

    async def get_relevant_context(
        self,
        question: str,
        max_entities: int = 20,
        max_relationships: int = 30,
    ) -> str:
        """
        Build RAG context from raw KG data (no LLM generation).

        Optimized paths:
          - Neo4j backend: single Cypher query filters server-side.
          - NetworkX backend: fast in-memory keyword match with index.

        Returns:
            Structured string of entities + relationships, or "" if nothing found.
        """
        # -- 1. Extract keywords from question --
        raw_tokens = question.lower().split()
        keywords = set()
        for token in raw_tokens:
            cleaned = token.strip(".,?!:;\"'()[]{}").lower()
            if len(cleaned) >= 2:
                keywords.add(cleaned)

        if not keywords:
            return ""

        backend = settings.NEXUSRAG_KG_GRAPH_BACKEND.lower()

        if backend == "neo4j":
            return await self._get_relevant_context_neo4j(
                keywords, max_entities, max_relationships,
            )
        else:
            return await self._get_relevant_context_networkx(
                keywords, max_entities, max_relationships,
            )

    async def _get_relevant_context_neo4j(
        self,
        keywords: set[str],
        max_entities: int,
        max_relationships: int,
    ) -> str:
        """Optimized KG context retrieval using a single Cypher query."""
        import time as _time
        t0 = _time.time()

        try:
            from neo4j import AsyncGraphDatabase
        except ImportError:
            logger.warning("neo4j driver not installed — falling back to generic path")
            return await self._get_relevant_context_networkx(
                keywords, max_entities, max_relationships,
            )

        label = f"kb_{self.workspace_id}"

        # Build a Cypher WHERE clause that does fuzzy substring matching on
        # entity names.  toLower(n.entity_id) CONTAINS $kw for each keyword,
        # combined with OR.
        where_parts = []
        params: dict = {}
        for i, kw in enumerate(keywords):
            param_name = f"kw{i}"
            where_parts.append(f"toLower(n.entity_id) CONTAINS ${param_name}")
            params[param_name] = kw

        where_clause = " OR ".join(where_parts)

        # One round-trip: find matching nodes + their 1-hop relationships
        cypher = f"""
        MATCH (n:`{label}`)
        WHERE {where_clause}
        WITH n LIMIT {max_entities}
        OPTIONAL MATCH (n)-[r]-(m:`{label}`)
        RETURN
            n.entity_id     AS entity_name,
            n.entity_type   AS entity_type,
            n.description   AS entity_desc,
            type(r)          AS rel_type,
            r.description    AS rel_desc,
            r.keywords       AS rel_kw,
            startNode(r).entity_id AS rel_src,
            endNode(r).entity_id   AS rel_tgt
        LIMIT {max_entities + max_relationships}
        """

        entity_info: dict[str, dict] = {}
        relevant_rels: list[dict] = []

        try:
            async with AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
            ) as driver:
                async with driver.session() as session:
                    result = await session.run(cypher, **params)
                    records = await result.data()

            for rec in records:
                ename = rec.get("entity_name", "")
                if ename and ename not in entity_info:
                    entity_info[ename] = {
                        "entity_type": rec.get("entity_type", "Unknown"),
                        "description": rec.get("entity_desc", ""),
                    }
                rel_src = rec.get("rel_src")
                rel_tgt = rec.get("rel_tgt")
                if rel_src and rel_tgt and len(relevant_rels) < max_relationships:
                    relevant_rels.append({
                        "source": rel_src,
                        "target": rel_tgt,
                        "description": rec.get("rel_desc", ""),
                        "keywords": rec.get("rel_kw", ""),
                    })
                    # Enrich connected entities
                    for connected in (rel_src, rel_tgt):
                        if connected and connected not in entity_info:
                            entity_info[connected] = {
                                "entity_type": "Unknown",
                                "description": "",
                            }

        except Exception as e:
            logger.error(
                f"Neo4j KG context query failed for workspace {self.workspace_id}: {e}"
            )
            return ""

        elapsed_ms = int((_time.time() - t0) * 1000)
        matched_list = list(entity_info.keys())[:max_entities]
        result = self._format_kg_context(matched_list, entity_info, relevant_rels)
        logger.info(
            f"KG raw context (Neo4j): {len(matched_list)} entities, "
            f"{len(relevant_rels)} rels for workspace {self.workspace_id} "
            f"in {elapsed_ms}ms"
        )
        return result

    async def _get_relevant_context_networkx(
        self,
        keywords: set[str],
        max_entities: int,
        max_relationships: int,
    ) -> str:
        """Optimized KG context retrieval for NetworkX (in-memory) backend."""
        import time as _time
        t0 = _time.time()

        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            all_nodes = await storage.get_all_nodes()
            all_edges = await storage.get_all_edges()
        except Exception as e:
            logger.error(f"Failed to get raw KG data for workspace {self.workspace_id}: {e}")
            return ""

        if not all_nodes:
            return ""

        # -- Build an O(1) lookup index: id → node --
        node_index: dict[str, dict] = {n.get("id", ""): n for n in all_nodes}

        # -- Find matching entities --
        matched_entity_names: set[str] = set()
        entity_info: dict[str, dict] = {}

        for node_id, node in node_index.items():
            node_lower = node_id.lower()
            matched = False
            for kw in keywords:
                if kw in node_lower or node_lower in kw:
                    matched = True
                    break
                for part in node_lower.split("-"):
                    if kw in part or part in kw:
                        matched = True
                        break
                if matched:
                    break

            if matched:
                matched_entity_names.add(node_id)
                entity_info[node_id] = {
                    "entity_type": node.get("entity_type", "Unknown"),
                    "description": node.get("description", ""),
                }

        if not matched_entity_names and len(all_nodes) <= 50:
            for node in all_nodes[:10]:
                nid = node.get("id", "")
                matched_entity_names.add(nid)
                entity_info[nid] = {
                    "entity_type": node.get("entity_type", "Unknown"),
                    "description": node.get("description", ""),
                }

        if not matched_entity_names:
            return ""

        matched_list = list(matched_entity_names)[:max_entities]

        # -- Find relationships using O(1) index instead of inner loop --
        relevant_rels: list[dict] = []
        matched_lower = {n.lower() for n in matched_list}

        for edge in all_edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src.lower() in matched_lower or tgt.lower() in matched_lower:
                relevant_rels.append({
                    "source": src,
                    "target": tgt,
                    "description": edge.get("description", ""),
                    "keywords": edge.get("keywords", ""),
                })
                # Use index for O(1) lookups instead of inner loop
                for connected in (src, tgt):
                    if connected not in entity_info and connected in node_index:
                        n = node_index[connected]
                        entity_info[connected] = {
                            "entity_type": n.get("entity_type", "Unknown"),
                            "description": n.get("description", ""),
                        }

            if len(relevant_rels) >= max_relationships:
                break

        elapsed_ms = int((_time.time() - t0) * 1000)
        result = self._format_kg_context(matched_list, entity_info, relevant_rels)
        logger.info(
            f"KG raw context (NetworkX): {len(matched_list)} entities, "
            f"{len(relevant_rels)} rels for workspace {self.workspace_id} "
            f"in {elapsed_ms}ms"
        )
        return result

    @staticmethod
    def _format_kg_context(
        matched_list: list[str],
        entity_info: dict[str, dict],
        relevant_rels: list[dict],
    ) -> str:
        """Format matched entities and relationships as structured text."""
        parts: list[str] = []

        if matched_list:
            parts.append("Entities found in documents:")
            for name in matched_list:
                info = entity_info.get(name, {})
                etype = info.get("entity_type", "")
                desc = info.get("description", "")
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                type_str = f" [{etype}]" if etype and etype != "Unknown" else ""
                if desc:
                    parts.append(f"- {name}{type_str}: {desc}")
                else:
                    parts.append(f"- {name}{type_str}")

        if relevant_rels:
            parts.append("")
            parts.append("Relationships:")
            for rel in relevant_rels:
                desc = rel["description"]
                if len(desc) > 150:
                    desc = desc[:150] + "..."
                if desc:
                    parts.append(f"- {rel['source']} → {rel['target']}: {desc}")
                else:
                    parts.append(f"- {rel['source']} → {rel['target']}")

        return "\n".join(parts)

    async def get_analytics(self) -> dict:
        """
        Compute KG analytics summary.

        Returns: entity_count, relationship_count, entity_types, top_entities, avg_degree.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            all_nodes = await storage.get_all_nodes()
            all_edges = await storage.get_all_edges()
        except Exception as e:
            logger.error(f"Failed to get KG analytics for workspace {self.workspace_id}: {e}")
            return {
                "entity_count": 0,
                "relationship_count": 0,
                "entity_types": {},
                "top_entities": [],
                "avg_degree": 0.0,
            }

        entity_count = len(all_nodes)
        relationship_count = len(all_edges)

        # Count entity types
        type_counts: dict[str, int] = {}
        entities_with_degree = []
        for node in all_nodes:
            etype = node.get("entity_type", "Unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1
            node_id = node.get("id") or node.get("entity_id") or ""
            try:
                degree = await storage.node_degree(node_id) if node_id else 0
            except Exception:
                degree = 0
            entities_with_degree.append({
                "name": node.get("id", ""),
                "entity_type": etype,
                "description": node.get("description", ""),
                "degree": degree,
            })

        # Sort by degree for top entities
        entities_with_degree.sort(key=lambda e: e["degree"], reverse=True)
        top_entities = entities_with_degree[:10]

        avg_degree = (
            sum(e["degree"] for e in entities_with_degree) / entity_count
            if entity_count > 0
            else 0.0
        )

        return {
            "entity_count": entity_count,
            "relationship_count": relationship_count,
            "entity_types": type_counts,
            "top_entities": top_entities,
            "avg_degree": round(avg_degree, 2),
        }
