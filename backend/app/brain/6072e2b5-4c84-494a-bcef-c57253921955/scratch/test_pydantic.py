import uuid
from typing import Annotated, Literal, TypedDict
import operator
from pydantic import BaseModel, ValidationError

class ChatSourceChunk(BaseModel):
    index: str
    chunk_id: str
    content: str
    document_id: uuid.UUID
    page_no: int = 0
    heading_path: list[str] = []
    score: float = 0.0
    source_type: str = "vector"
    source_file: str | None = None

class SupervisorState(TypedDict):
    sources: Annotated[list[ChatSourceChunk], operator.add]

def test():
    # Test valid
    chunk = ChatSourceChunk(
        index="id1",
        chunk_id="c1",
        content="hello",
        document_id=uuid.uuid4()
    )
    print("Valid chunk created")
    
    # Test invalid (missing content)
    try:
        ChatSourceChunk(
            index="id2",
            chunk_id="c2",
            document_id=uuid.uuid4()
        )
    except ValidationError as e:
        print(f"Pydantic Error caught: {e}")

if __name__ == "__main__":
    test()
