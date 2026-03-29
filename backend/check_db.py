import asyncio
from sqlalchemy import text
from app.core.database import engine

async def run():
    async with engine.begin() as conn:
        res = await conn.execute(text("SELECT id, original_filename, document_number, location, issuing_agency, parent_agency, published_date FROM documents ORDER BY id DESC LIMIT 5"))
        for row in res.fetchall():
            print(dict(row._mapping))

asyncio.run(run())
