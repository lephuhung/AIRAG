import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import async_session_maker
from app.models.user import User
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from sqlalchemy import select, func

async def test():
    try:
        async with async_session_maker() as db:
            users = await db.scalar(select(func.count(User.id)))
            docs = await db.scalar(select(func.count(Document.id)))
            kbs = await db.scalar(select(func.count(KnowledgeBase.id)))
            print(f"Users: {users}, Docs: {docs}, KBs: {kbs}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test())
