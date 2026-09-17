import sys; sys.path.insert(0, '.')
from src.tools.knowledge_base import _embed_with_retry, _get_qdrant, COLLECTION_NAME

query = 'What is CNN accuracy in medical imaging?'
emb = _embed_with_retry(query)
try:
    qdrant = _get_qdrant()
    results = qdrant.query_points(collection_name=COLLECTION_NAME, query=emb, limit=5, with_payload=True)
    print('Raw Qdrant results for query:', query)
    for r in results.points:
        print(f"Score: {r.score:.4f}, Text: {r.payload.get('text', '')[:50]}...")
except Exception as e:
    print("Qdrant error:", e)
