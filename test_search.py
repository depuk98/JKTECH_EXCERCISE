import asyncio
from app.services.document import DocumentService
from app.db.session import SessionLocal

async def test_search():
    """Test the search functionality."""
    db = SessionLocal()
    try:
        print("Testing search functionality...")
        results = await DocumentService.search_documents(
            db=db, 
            user_id=1, 
            query='test query', 
            limit=5
        )
        print(f'Found {len(results)} results')
        
        if results:
            print("\nTop results:")
            for i, (chunk, score) in enumerate(results[:3], 1):
                print(f"{i}. Score: {score:.4f}")
                print(f"   Text: {chunk.text[:100]}...")
                print(f"   Metadata: {chunk.chunk_metadata}")
                print()
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(test_search()) 