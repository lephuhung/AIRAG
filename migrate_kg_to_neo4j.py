"""
migrate_kg_to_neo4j.py
========================
One-shot script to migrate existing NetworkX / graphml KG data into Neo4j
for all workspaces that have a backend/data/lightrag/kb_<N>/ directory.

Usage (from repo root, with venv active):
    python migrate_kg_to_neo4j.py              # migrate all workspaces
    python migrate_kg_to_neo4j.py --ws 1       # only workspace 1
    python migrate_kg_to_neo4j.py --dry-run    # show what would be migrated

The script uses raw Cypher (same queries as LightRAG's Neo4JStorage.upsert_node /
upsert_edge) so it does NOT need LightRAG to be importable.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate repo root and load .env so we can read NEO4J_* vars
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
os.chdir(REPO_ROOT / "backend")  # LightRAG's dotenv loads ".env" from cwd

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env", override=False)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.environ.get("NEO4J_PASSWORD", "nexusrag123")

LIGHTRAG_DATA_DIR = REPO_ROOT / "backend" / "data" / "lightrag"
NS = "http://graphml.graphdrawing.org/xmlns"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_graphml(graphml_path: Path) -> tuple[list[dict], list[dict]]:
    """Parse a graphml file and return (nodes, edges) as plain dicts."""
    tree = ET.parse(graphml_path)
    root = tree.getroot()

    # Build key_id -> attr_name map
    key_map: dict[str, str] = {}
    for key in root.findall(f"{{{NS}}}key"):
        key_map[key.get("id", "")] = key.get("attr.name", "")

    graph = root.find(f"{{{NS}}}graph")
    if graph is None:
        return [], []

    nodes: list[dict] = []
    for node_el in graph.findall(f"{{{NS}}}node"):
        node_id = node_el.get("id", "")
        props: dict[str, str] = {"entity_id": node_id}
        for d in node_el.findall(f"{{{NS}}}data"):
            attr_name = key_map.get(d.get("key", ""), "")
            if attr_name and d.text is not None:
                props[attr_name] = d.text
        # Ensure entity_id is set (may already be set via d0 key)
        props.setdefault("entity_id", node_id)
        nodes.append(props)

    edges: list[dict] = []
    for edge_el in graph.findall(f"{{{NS}}}edge"):
        src = edge_el.get("source", "")
        tgt = edge_el.get("target", "")
        props = {}
        for d in edge_el.findall(f"{{{NS}}}data"):
            attr_name = key_map.get(d.get("key", ""), "")
            if attr_name and d.text is not None:
                props[attr_name] = d.text
        edges.append({"source": src, "target": tgt, **props})

    return nodes, edges


async def migrate_workspace(driver, workspace_id: int, dry_run: bool) -> None:
    """Migrate one workspace's graphml data into Neo4j."""
    graphml = LIGHTRAG_DATA_DIR / f"kb_{workspace_id}" / "graph_chunk_entity_relation.graphml"
    if not graphml.exists():
        print(f"  [ws={workspace_id}] No graphml found at {graphml} — skipping")
        return

    nodes, edges = parse_graphml(graphml)
    label = f"kb_{workspace_id}"

    print(f"  [ws={workspace_id}] Found {len(nodes)} nodes, {len(edges)} edges in graphml")

    if dry_run:
        for n in nodes[:3]:
            print(f"    NODE: {n.get('entity_id')} [{n.get('entity_type')}]")
        for e in edges[:3]:
            print(f"    EDGE: {e['source']} → {e['target']}")
        print("    (dry-run: no writes)")
        return

    # Ensure index exists
    async with driver.session() as session:
        try:
            await session.run(
                f"CREATE INDEX IF NOT EXISTS FOR (n:`{label}`) ON (n.entity_id)"
            )
        except Exception as e:
            print(f"  [ws={workspace_id}] Index creation warning: {e}")

    # Upsert nodes
    node_ok = 0
    node_err = 0
    async with driver.session() as session:
        for node in nodes:
            entity_id = node.get("entity_id", "")
            entity_type = node.get("entity_type", "UNKNOWN") or "UNKNOWN"
            # Sanitize entity_type (same as LightRAG)
            entity_type = entity_type.replace("`", "").strip()
            if "," in entity_type:
                entity_type = entity_type.split(",")[0].strip()
            if not entity_type:
                entity_type = "UNKNOWN"
            node["entity_type"] = entity_type

            try:
                query = f"""
                MERGE (n:`{label}` {{entity_id: $entity_id}})
                SET n += $properties
                SET n:`{entity_type}`
                """
                await session.run(query, entity_id=entity_id, properties=node)
                node_ok += 1
            except Exception as e:
                print(f"  [ws={workspace_id}] Node upsert error ({entity_id}): {e}")
                node_err += 1

    print(f"  [ws={workspace_id}] Nodes: {node_ok} upserted, {node_err} errors")

    # Upsert edges
    edge_ok = 0
    edge_err = 0
    async with driver.session() as session:
        for edge in edges:
            src = edge["source"]
            tgt = edge["target"]
            # Build edge properties (exclude source/target keys)
            props = {k: v for k, v in edge.items() if k not in ("source", "target")}

            try:
                query = f"""
                MATCH (source:`{label}` {{entity_id: $source_id}})
                WITH source
                MATCH (target:`{label}` {{entity_id: $target_id}})
                MERGE (source)-[r:DIRECTED]-(target)
                SET r += $properties
                RETURN r
                """
                result = await session.run(
                    query, source_id=src, target_id=tgt, properties=props
                )
                await result.consume()
                edge_ok += 1
            except Exception as e:
                print(f"  [ws={workspace_id}] Edge upsert error ({src} → {tgt}): {e}")
                edge_err += 1

    print(f"  [ws={workspace_id}] Edges: {edge_ok} upserted, {edge_err} errors")

    # Verify
    async with driver.session() as session:
        result = await session.run(f"MATCH (n:`{label}`) RETURN count(n) as cnt")
        record = await result.single()
        count = record["cnt"] if record else 0
        print(f"  [ws={workspace_id}] Neo4j now has {count} nodes with label {label}")


async def main(workspace_ids: list[int] | None, dry_run: bool) -> None:
    from neo4j import AsyncGraphDatabase

    print(f"Connecting to Neo4j: {NEO4J_URI}")
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    try:
        # Discover workspaces
        if workspace_ids:
            ws_list = workspace_ids
        else:
            ws_list = []
            if LIGHTRAG_DATA_DIR.exists():
                for d in sorted(LIGHTRAG_DATA_DIR.iterdir()):
                    if d.is_dir() and d.name.startswith("kb_"):
                        try:
                            ws_list.append(int(d.name[3:]))
                        except ValueError:
                            pass
            if not ws_list:
                print("No workspace directories found under backend/data/lightrag/")
                return

        print(f"Workspaces to migrate: {ws_list}")
        for ws_id in ws_list:
            print(f"\n--- Workspace {ws_id} ---")
            await migrate_workspace(driver, ws_id, dry_run)

        print("\nMigration complete.")
    finally:
        await driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate NetworkX KG data to Neo4j")
    parser.add_argument("--ws", type=int, nargs="+", help="Workspace IDs to migrate (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without writing")
    args = parser.parse_args()

    asyncio.run(main(args.ws, args.dry_run))
