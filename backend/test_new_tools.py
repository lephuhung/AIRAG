#!/usr/bin/env python3
"""
Test script for new LangGraph tools: search_documents_number and search_abbreviation
"""

import asyncio
import os

os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5433/nexusrag"
)


async def main():
    from app.services.agent.tools import search_documents_number, search_abbreviation
    from app.core.db import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        # Test 1: search_documents_number
        print("\n" + "=" * 60)
        print("TEST 1: search_documents_number")
        print("=" * 60)

        result1 = await search_documents_number(
            query="2024",
            workspace_ids=[1],
            db=db,
        )
        print(f"Result: {result1}")

        # Test 2: search_abbreviation (found)
        print("\n" + "=" * 60)
        print("TEST 2: search_abbreviation (search for 'AI')")
        print("=" * 60)

        result2 = await search_abbreviation(
            abbreviation="AI",
            workspace_ids=[1],
            db=db,
        )
        print(f"Result: {result2}")

        # Test 3: search_abbreviation (not found - ask for clarification)
        print("\n" + "=" * 60)
        print("TEST 3: search_abbreviation (not found - XYZ123)")
        print("=" * 60)

        result3 = await search_abbreviation(
            abbreviation="XYZ123",
            workspace_ids=[1],
            db=db,
        )
        print(f"Result: {result3}")


if __name__ == "__main__":
    asyncio.run(main())
