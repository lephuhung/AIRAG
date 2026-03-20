import os
import sys
import asyncio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import async_session_maker
from app.api.admin import list_users
from app.models.user import User
from sqlalchemy import select

async def test_list_users():
    try:
        async with async_session_maker() as db:
            user = await db.scalar(select(User).limit(1))
            res = await list_users(search=None, is_active=None, page=1, per_page=20, db=db, current_user=user)
            print(f"Success! Found {res.total} users.")
            print(f"Users: {[u.email for u in res.users]}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_list_users())
