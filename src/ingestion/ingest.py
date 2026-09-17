"""
ingest.py
---------
Document ingestion pipeline for the Agentic RAG system.

Reads PDF and TXT files from /data, chunks them, embeds each chunk
with gemini-embedding-001 (output_dimensionality=768), and upserts into
a Qdrant collection named "private_knowledge".

Usage:
    python src/ingestion/ingest.py                  # ingest all docs in /data
    python src/ingestion/ingest.py --test "query"   # run retrieval test only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Generator

import pdfplumber
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types as genai_types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

# Load .env — walk up directory tree to find it, override any stale system vars
load_dotenv(find_dotenv(usecwd=True), override=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLLECTION_NAME = "private_knowledge"
CHUNK_SIZE = 800        # characters (approx 200 tokens; splitter uses chars)
CHUNK_OVERLAP = 80      # characters overlap between consecutive chunks
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768     # MRL-reduced — must match output_dimensionality below
EMBED_BATCH_SIZE = 5    # keep well within free-tier RPM

# Retry settings for 429s
MAX_RETRY_ATTEMPTS = 5
BASE_DELAY = 1.0
BACKOFF_FACTOR = 2.0

# ---------------------------------------------------------------------------
# Clients (lazy — created once per process)
# ---------------------------------------------------------------------------
_gemini_client: genai.Client | None = None
_qdrant_client: QdrantClient | None = None


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Copy .env.template to .env and fill in your key."
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None
        _qdrant_client = QdrantClient(url=url, api_key=api_key)
        logger.info("Connected to Qdrant at %s", url)
    return _qdrant_client


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def embed_with_retry(
    client: genai.Client,
    text: str,
    task_type: str,
) -> list[float]:
    """
    Embed a single text string, retrying on 429 with exponential backoff.
    task_type: "RETRIEVAL_DOCUMENT" for ingestion, "RETRIEVAL_QUERY" for search.
    """
    delay = BASE_DELAY
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config=genai_types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
            return result.embeddings[0].values
        except Exception as exc:
            is_rate_limit = (
                "429" in str(exc)
                or "RESOURCE_EXHAUSTED" in str(exc)
                or getattr(exc, "status_code", None) == 429
            )
            if is_rate_limit and attempt < MAX_RETRY_ATTEMPTS:
                logger.warning(
                    "Rate limit hit. Attempt %d/%d — retrying in %.1fs",
                    attempt, MAX_RETRY_ATTEMPTS, delay,
                )
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
            else:
                raise
    raise RuntimeError("Exhausted retry attempts for embedding")


# ---------------------------------------------------------------------------
# Step 1: Load documents
# ---------------------------------------------------------------------------

def _read_pdf(path: Path) -> Generator[dict, None, None]:
    """Yield one dict per page from a PDF file."""
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                yield {
                    "text": text,
                    "source_filename": path.name,
                    "page_number": page_num,
                }


def _read_txt(path: Path) -> Generator[dict, None, None]:
    """Yield a single dict for the entire TXT file (page_number=1)."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if text:
        yield {
            "text": text,
            "source_filename": path.name,
            "page_number": 1,
        }


def load_documents(data_dir: str) -> list[dict]:
    """
    Walk data_dir and load every .pdf and .txt file.
    Returns a list of page-level dicts with keys:
        text, source_filename, page_number
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    docs: list[dict] = []
    files = sorted(data_path.glob("**/*"))
    supported = {".pdf", ".txt"}

    for file in files:
        if file.suffix.lower() not in supported:
            continue
        if file.name.lower() == "readme.txt":
            continue  # skip the /data placeholder file
        logger.info("Loading: %s", file.name)
        try:
            if file.suffix.lower() == ".pdf":
                docs.extend(_read_pdf(file))
            else:
                docs.extend(_read_txt(file))
        except Exception as exc:
            logger.error("Failed to read %s: %s", file.name, exc)

    logger.info("Loaded %d page(s) from %s", len(docs), data_dir)
    return docs


# ---------------------------------------------------------------------------
# Step 2: Chunk documents
# ---------------------------------------------------------------------------

def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split page texts with RecursiveCharacterTextSplitter.
    Each chunk inherits source_filename + page_number and gets a unique chunk_id.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    chunk_counter = 0

    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for split_text in splits:
            split_text = split_text.strip()
            if not split_text:
                continue
            chunks.append(
                {
                    "text": split_text,
                    "source_filename": doc["source_filename"],
                    "page_number": doc["page_number"],
                    "chunk_id": chunk_counter,
                }
            )
            chunk_counter += 1

    logger.info("Created %d chunk(s) from %d page(s)", len(chunks), len(documents))
    return chunks


# ---------------------------------------------------------------------------
# Step 3: Embed chunks
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed each chunk with task_type=RETRIEVAL_DOCUMENT.
    Adds a 'vector' key (list[float], length 768) to each chunk dict.
    Batches requests to stay within free-tier RPM.
    """
    client = get_gemini_client()
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        logger.info("Embedding chunk %d/%d — %s p.%s",
                    i + 1, total,
                    chunk["source_filename"],
                    chunk["page_number"])

        chunk["vector"] = embed_with_retry(
            client,
            chunk["text"],
            task_type="RETRIEVAL_DOCUMENT",
        )

        # Polite inter-request pause to stay under free-tier RPM
        if (i + 1) % EMBED_BATCH_SIZE == 0 and i + 1 < total:
            logger.info("  Pausing 3s to respect free-tier RPM...")
            time.sleep(3)

    logger.info("Embedding complete.")
    return chunks


# ---------------------------------------------------------------------------
# Step 4: Upsert to Qdrant
# ---------------------------------------------------------------------------

def ensure_collection(client: QdrantClient) -> None:
    """Create the Qdrant collection if it doesn't already exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection '%s' (dim=%d, cosine)", 
                    COLLECTION_NAME, EMBEDDING_DIM)
    else:
        logger.info("Collection '%s' already exists — upserting into it.", COLLECTION_NAME)


def upsert_to_qdrant(chunks: list[dict]) -> None:
    """
    Upsert all embedded chunks into Qdrant.
    Each point payload includes: text, source_filename, page_number, chunk_id.
    """
    client = get_qdrant_client()
    ensure_collection(client)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=chunk["vector"],
            payload={
                "text": chunk["text"],
                "source_filename": chunk["source_filename"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
            },
        )
        for chunk in chunks
    ]

    # Upsert in batches of 100
    batch_size = 100
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        logger.info("Upserted points %d–%d / %d", 
                    start + 1, start + len(batch), len(points))

    logger.info("All %d chunks upserted to '%s'.", len(points), COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Retrieval test (no LLM — pure vector search)
# ---------------------------------------------------------------------------

def retrieval_test(query: str, top_k: int = 5) -> None:
    """
    Embed a query with task_type=RETRIEVAL_QUERY and print the top_k
    Qdrant results with similarity_score, source_filename, page_number.
    """
    logger.info("Running retrieval test for query: %r", query)

    gemini = get_gemini_client()
    query_vector = embed_with_retry(gemini, query, task_type="RETRIEVAL_QUERY")

    qdrant = get_qdrant_client()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    hits = results.points
    print(f"\n{'='*60}")
    print(f"Query: {query!r}")
    print(f"Top {len(hits)} result(s):")
    print(f"{'='*60}")

    if not hits:
        print("  (no results returned — is the collection populated?)")
        return

    for rank, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        print(f"\n[{rank}] Score: {hit.score:.4f}")
        print(f"    Source : {payload.get('source_filename', 'N/A')}")
        print(f"    Page   : {payload.get('page_number', 'N/A')}")
        print(f"    Chunk  : {payload.get('chunk_id', 'N/A')}")
        snippet = payload.get("text", "")[:300].replace("\n", " ")
        print(f"    Text   : {snippet}...")

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run_ingestion(data_dir: str) -> None:
    """Full pipeline: load → chunk → embed → upsert."""
    logger.info("=== Ingestion pipeline starting ===")
    docs = load_documents(data_dir)
    if not docs:
        logger.warning("No documents found in %s. Place PDFs or TXTs there first.", data_dir)
        return
    chunks = chunk_documents(docs)
    chunks = embed_chunks(chunks)
    upsert_to_qdrant(chunks)
    logger.info("=== Ingestion complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic RAG ingestion pipeline")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "..", "data"),
        help="Directory containing PDFs/TXTs to ingest (default: ./data)",
    )
    parser.add_argument(
        "--test",
        metavar="QUERY",
        default=None,
        help="Skip ingestion and run a retrieval test with this query string",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return in retrieval test (default: 5)",
    )
    args = parser.parse_args()

    if args.test:
        retrieval_test(args.test, top_k=args.top_k)
    else:
        run_ingestion(args.data_dir)
