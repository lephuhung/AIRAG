import os
import sys
import asyncio
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.main import app
from app.core.database import async_session_maker
from app.models.user import User
from sqlalchemy import select

async def get_super_admin():
    async with async_session_maker() as db:
        user = await db.scalar(select(User).where(User.is_superadmin == True).limit(1))
        return user

def run_test():
    loop = asyncio.get_event_loop()
    user = loop.run_until_complete(get_super_admin())
    if not user:
        print("No superadmin found!")
        return
        
    # We will patch the require_superadmin dependency
    from app.core.deps import require_superadmin
    app.dependency_overrides[require_superadmin] = lambda: user
    
    client = TestClient(app)
    
    print("Testing /api/v1/admin/stats ...")
    res = client.get("/api/v1/admin/stats")
    print(f"Stats Status: {res.status_code}")
    if res.status_code != 200:
        print(f"Stats Error: {res.text}")
        
    print("Testing /api/v1/admin/users ...")
    res = client.get("/api/v1/admin/users")
    print(f"Users Status: {res.status_code}")
    if res.status_code != 200:
        print(f"Users Error: {res.text}")
    else:
        print(f"Users Output length: len(res.json().get('users', []))")

if __name__ == "__main__":
    run_test()
