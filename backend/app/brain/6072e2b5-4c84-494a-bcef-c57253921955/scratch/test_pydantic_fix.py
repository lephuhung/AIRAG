import uuid
import json
from pydantic import ValidationError
from app.schemas.rag import PersistedChatMessage, ChatSourceChunk

def test_validation():
    # 1. Test ChatSourceChunk with string heading_path
    bad_source = {
        "index": "id1",
        "chunk_id": "chunk1",
        "content": "some content",
        "document_id": str(uuid.uuid4()),
        "page_no": 1,
        "heading_path": "Chương I > Điều 1", # String instead of list
        "score": 0.9,
        "source_type": "vector",
        "source_file": "test.pdf"
    }
    
    try:
        source = ChatSourceChunk.model_validate(bad_source)
        print("ChatSourceChunk validation successful!")
        print(f"Heading path: {source.heading_path}")
    except ValidationError as e:
        print(f"ChatSourceChunk validation failed: {e}")

    # 2. Test PersistedChatMessage with mixed data
    bad_msg = {
        "id": uuid.uuid4(),
        "message_id": "msg1",
        "role": "assistant",
        "content": "hello",
        "document_ids": [str(uuid.uuid4()), str(uuid.uuid4())], # String UUIDs
        "sources": [bad_source],
        "created_at": "2024-05-09T12:00:00Z"
    }
    
    try:
        msg = PersistedChatMessage.model_validate(bad_msg)
        print("PersistedChatMessage validation successful!")
        print(f"Doc IDs type: {type(msg.document_ids[0])}")
        print(f"Sources[0] heading_path type: {type(msg.sources[0].heading_path)}")
    except ValidationError as e:
        print(f"PersistedChatMessage validation failed: {e}")

if __name__ == "__main__":
    test_validation()
