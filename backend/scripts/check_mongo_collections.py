"""
Script kiểm tra cấu trúc thực tế của các MongoDB collections.
Chạy: python -m scripts.check_mongo_collections

In ra TẤT CẢ các fields của mỗi document mẫu (5 docs đầu tiên).
Dùng để xác nhận field names thực tế trong MongoDB.
"""

import asyncio
from app.core.config import settings


async def check_collections():
    from app.services.mongo_client import get_mongo_client

    client = get_mongo_client()
    db = client[settings.MONGO_DATABASE]

    collections_to_check = ["bhxh", "evn", "lg", "vacxin", "cv19", "uids", "vnvc"]

    for col_name in collections_to_check:
        col = db[col_name]
        print(f"\n{'='*60}")
        print(f"COLLECTION: {col_name}")
        print(f"{'='*60}")

        # Count total
        count = await col.count_documents({})
        print(f"Total documents: {count}")

        # Sample 3 docs — show ALL fields
        cursor = col.find({}).limit(3)
        docs = await cursor.to_list(length=3)

        for i, doc in enumerate(docs, 1):
            doc_id = doc.pop("_id", "N/A")
            print(f"\n  [-- Doc {i} (id={doc_id}) --]")
            if not doc:
                print("    (empty document)")
                continue
            for key, val in doc.items():
                val_preview = repr(val)[:80]
                print(f"    {key}: {val_preview}")


if __name__ == "__main__":
    asyncio.run(check_collections())
