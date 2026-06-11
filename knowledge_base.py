"""
Knowledge Base Engine — document ingestion, chunking, embedding, and RAG retrieval.

Supports: .md, .txt, .pdf, .json
Embedding model: BAAI/bge-small-zh-v1.5
Vector store: ChromaDB (collection: knowledge_docs)
Search: Hybrid BM25 + Vector with RRF fusion
"""

import os
import uuid
import re
import json
import hashlib
from datetime import datetime, timezone
from typing import Optional, TypedDict

import chromadb
from chromadb.api.types import EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from storage import atomic_write_json, load_json


# ─── ChromaDB Embedding Function ─────────────────────────────

class BGEChromaEmbeddingFunction(EmbeddingFunction):
    """ChromaDB embedding function wrapping BAAI/bge-small-zh-v1.5."""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: list[str]) -> Embeddings:
        embeddings = self.model.encode(input, show_progress_bar=False)
        return embeddings.tolist()


# ─── Document Info Type ──────────────────────────────────────

class DocumentInfo(TypedDict):
    source: str
    file_path: str
    file_hash: str
    chunk_count: int
    imported_at: str


# ─── Knowledge Base ──────────────────────────────────────────

class KnowledgeBase:
    """Manages document import, storage, and hybrid search."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "knowledge_docs",
        model_name: str = "BAAI/bge-small-zh-v1.5",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._model_name = model_name

        # Embedding model (manual path for batch control)
        self.embedder = SentenceTransformer(model_name)
        self._dimension = self.embedder.get_sentence_embedding_dimension()

        # ChromaDB embedding function (registered as fallback)
        self._embed_fn = BGEChromaEmbeddingFunction(model_name)

        # ChromaDB
        os.makedirs(persist_directory, exist_ok=True)
        self._persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
        except ValueError:
            # Existing collection has a different embedding function; get it as-is
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        # Document manifest
        self._manifest_path = os.path.join(persist_directory, "kb_manifest.json")
        self._manifest: dict[str, DocumentInfo] = {}
        self._load_manifest()

        # BM25 index
        self._bm25_index: Optional[BM25Okapi] = None
        self._tokenized_corpus: list[list[str]] = []
        self._bm25_doc_ids: list[str] = []
        self._rebuild_bm25_index()

    # ─── Document CRUD ──────────────────────────────────────────

    def load_document(self, path: str, force_reload: bool = False) -> int:
        """Import a document file into the knowledge base.

        Args:
            path: File path to import.
            force_reload: If True, re-import even if the file hash matches.

        Returns:
            Number of chunks stored (0 if skipped due to dedup).
        """
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Document not found: {path}")

        source = os.path.basename(path)

        # Compute file hash for dedup
        with open(path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # Dedup check
        existing = self._manifest.get(source)
        if existing and existing["file_hash"] == file_hash and not force_reload:
            return 0  # unchanged file, skip

        # If re-importing a changed file, delete old chunks first
        if existing:
            self.delete_document(source)

        # Extract text
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            text = self._extract_pdf(path)
        elif ext == ".json":
            text = self._extract_json(path)
        else:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

        if not text.strip():
            return 0

        chunks = self.chunk_text(text, source)
        self.store_chunks(chunks)

        # Update manifest
        self._manifest[source] = {
            "source": source,
            "file_path": path,
            "file_hash": file_hash,
            "chunk_count": len(chunks),
            "imported_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_manifest()

        # Rebuild BM25 index
        self._rebuild_bm25_index()

        return len(chunks)

    def add_document(self, text: str, source: str) -> int:
        """Import raw text as a document (no file required).

        Args:
            text: The text content to import.
            source: Source label (e.g. 'chat-note.md').

        Returns:
            Number of chunks stored.

        Raises:
            ValueError: If source already exists in the manifest.
        """
        text = text.strip()
        if not text:
            return 0
        if not source or not source.strip():
            raise ValueError("Source label cannot be empty.")
        source = source.strip()

        if source in self._manifest:
            raise ValueError(
                f"Document '{source}' already exists. "
                "Use a different source name, or delete the existing one first."
            )

        chunks = self.chunk_text(text, source)
        self.store_chunks(chunks)

        self._manifest[source] = {
            "source": source,
            "file_path": "",
            "file_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "chunk_count": len(chunks),
            "imported_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_manifest()
        self._rebuild_bm25_index()
        return len(chunks)

    def delete_document(self, source: str) -> bool:
        """Delete all chunks from a specific document source.

        Args:
            source: The source filename to delete (e.g. 'readme.md').

        Returns:
            True if chunks were removed, False if source not found.
        """
        if source not in self._manifest:
            return False

        # Get all chunk IDs for this source
        all_data = self.collection.get(
            where={"source": source},
            include=[],
        )
        if all_data["ids"]:
            self.collection.delete(ids=all_data["ids"])

        # Remove from manifest
        del self._manifest[source]
        self._save_manifest()

        # Rebuild BM25 index
        self._rebuild_bm25_index()

        return True

    def get_document_info(self, source: str) -> Optional[dict]:
        """Get metadata and chunk previews for a document.

        Returns dict with:
            - info: DocumentInfo from manifest
            - chunks: list of {"id", "index", "text", "text_preview"}
            - total_chunks: int
        """
        if source not in self._manifest:
            return None

        info = self._manifest[source]

        # Fetch all chunks for this source
        results = self.collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
            limit=100,
        )

        chunks = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                text = results["documents"][i] if results["documents"] else ""
                chunks.append({
                    "id": results["ids"][i],
                    "index": meta.get("chunk_index", i),
                    "text": text,
                    "text_preview": text[:200] + ("..." if len(text) > 200 else ""),
                })

        chunks.sort(key=lambda c: c["index"])

        return {
            "info": info,
            "chunks": chunks,
            "total_chunks": len(chunks),
        }

    def list_documents(self) -> list[dict]:
        """List all imported documents with metadata.

        Returns list sorted by import time (newest first).
        """
        docs = list(self._manifest.values())
        docs.sort(key=lambda d: d["imported_at"], reverse=True)
        return docs

    # ─── Chunking ───────────────────────────────────────────────

    def chunk_text(self, text: str, source: str = "unknown") -> list[dict]:
        """Split text into overlapping chunks using recursive paragraph / sentence boundaries.

        Returns list of dicts: [{"text": ..., "metadata": {...}}, ...]
        """
        text = text.strip()

        # Step 1: split by double newline (paragraph boundary)
        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

        chunks = []
        buffer = ""

        for para in paragraphs:
            if len(buffer + "\n\n" + para) <= self.chunk_size:
                buffer = (buffer + "\n\n" + para).strip() if buffer else para
                continue

            # Buffer is full — flush it and carry an overlap tail forward as
            # context for the next chunk. The tail is only ever prepended to
            # real content, never emitted as a chunk of its own.
            overlap = ""
            if buffer:
                chunks.append(buffer)
                overlap = buffer[-self.chunk_overlap:] if len(buffer) > self.chunk_overlap else ""

            # Step 2: if a single paragraph exceeds chunk_size, split further
            if len(para) > self.chunk_size:
                sub_chunks = self._split_long_text(para)
                if sub_chunks and overlap:
                    sub_chunks[0] = (overlap + "\n\n" + sub_chunks[0]).strip()
                # Emit all but the last sub-chunk; keep the last as the running
                # buffer so it can still merge with subsequent paragraphs.
                chunks.extend(sub_chunks[:-1])
                buffer = sub_chunks[-1] if sub_chunks else ""
            else:
                buffer = (overlap + "\n\n" + para).strip() if overlap else para

        # Flush remaining buffer
        if buffer:
            chunks.append(buffer)

        # Build metadata
        total = len(chunks)
        result = []
        for i, chunk in enumerate(chunks):
            result.append({
                "text": chunk.strip(),
                "metadata": {
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": total,
                },
            })
        return result

    def store_chunks(self, chunks: list[dict]) -> int:
        """Embed and upsert chunks into ChromaDB.

        Returns number of chunks stored.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [str(uuid.uuid4()) for _ in chunks]

        # Add timestamp to metadata
        now = datetime.now(timezone.utc).isoformat()
        for m in metadatas:
            m["imported_at"] = now

        # Batch embed (16 at a time to keep memory stable)
        all_embeddings = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.embedder.encode(batch, show_progress_bar=False).tolist()
            all_embeddings.extend(embeddings)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=all_embeddings,
        )
        return len(chunks)

    # ─── Search ─────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3, hybrid: bool = True) -> list[dict]:
        """Search the knowledge base using hybrid retrieval (BM25 + vector).

        Args:
            query: Search query.
            top_k: Number of results to return.
            hybrid: If True, use BM25 + vector RRF fusion.
                    If False, use pure vector search.

        Returns:
            List of {"text": str, "source": str, "score": float}
        """
        if self.collection.count() == 0:
            return []

        if not hybrid or self._bm25_index is None:
            return self._vector_search_only(query, top_k)

        # === Vector search ===
        query_embedding = self.embedder.encode([query], show_progress_bar=False).tolist()
        vector_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k * 3,  # fetch more for fusion
        )

        vector_scores: dict[str, float] = {}
        if vector_results["ids"] and vector_results["ids"][0]:
            for i in range(len(vector_results["ids"][0])):
                doc_id = vector_results["ids"][0][i]
                score = 1.0 - (vector_results["distances"][0][i] / 2.0)
                if score >= 0.5:
                    vector_scores[doc_id] = score

        # === BM25 search ===
        tokenized_query = self._tokenize(query)
        bm25_scores = self._bm25_index.get_scores(tokenized_query)

        bm25_results: dict[str, float] = {}
        for i, score in enumerate(bm25_scores):
            if score > 0:
                bm25_results[self._bm25_doc_ids[i]] = float(score)

        # === RRF merge ===
        K_RRF = 60

        def _rank_dict(d: dict[str, float]) -> dict[str, int]:
            sorted_items = sorted(d.items(), key=lambda x: -x[1])
            return {item[0]: rank + 1 for rank, item in enumerate(sorted_items)}

        vector_ranks = _rank_dict(vector_scores)
        bm25_ranks = _rank_dict(bm25_results)

        all_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        fused: list[tuple[str, float]] = []
        for doc_id in all_ids:
            v_score = 1.0 / (K_RRF + vector_ranks.get(doc_id, 999))
            b_score = 1.0 / (K_RRF + bm25_ranks.get(doc_id, 999))
            fused.append((doc_id, v_score + b_score))

        fused.sort(key=lambda x: -x[1])

        # === Fetch actual text for top-k results ===
        top_ids = [doc_id for doc_id, _ in fused[:top_k]]
        if not top_ids:
            return []

        detail = self.collection.get(ids=top_ids, include=["documents", "metadatas"])
        results = []
        id_to_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(fused[:top_k])}
        if detail["ids"]:
            for i in range(len(detail["ids"])):
                doc_id = detail["ids"][i]
                meta = detail["metadatas"][i] if detail["metadatas"] else {}
                text = detail["documents"][i] if detail["documents"] else ""
                results.append({
                    "text": text,
                    "source": meta.get("source", "unknown"),
                    "score": round(1.0 / (id_to_rank.get(doc_id, 999) + K_RRF / 10), 4),
                })

        return results

    def _vector_search_only(self, query: str, top_k: int = 3) -> list[dict]:
        """Pure vector search (original implementation)."""
        query_embedding = self.embedder.encode([query], show_progress_bar=False).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        output = []
        for i in range(len(results["ids"][0])):
            score = 1.0 - (results["distances"][0][i] / 2.0)
            if score < 0.5:
                continue
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            output.append({
                "text": results["documents"][0][i],
                "source": meta.get("source", "unknown"),
                "score": round(score, 4),
            })
        return output

    # ─── BM25 Index ─────────────────────────────────────────────

    def _rebuild_bm25_index(self):
        """Rebuild BM25 index from all current chunks."""
        all_data = self.collection.get(include=["documents"])
        if not all_data or not all_data["documents"]:
            self._bm25_index = None
            self._tokenized_corpus = []
            self._bm25_doc_ids = []
            return

        self._bm25_doc_ids = all_data["ids"]
        self._tokenized_corpus = [self._tokenize(doc) for doc in all_data["documents"]]
        self._bm25_index = BM25Okapi(self._tokenized_corpus)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25.

        Character-level for CJK, word-level for Latin text.
        """
        parts = re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text.lower())
        return parts

    # ─── Utility ────────────────────────────────────────────────

    def count(self) -> int:
        """Return number of chunks stored in the knowledge base."""
        return self.collection.count()

    def list_sources(self) -> list[str]:
        """List unique document sources in the knowledge base."""
        all_meta = self.collection.get(include=["metadatas"])
        if not all_meta["metadatas"]:
            return []
        sources = set()
        for m in all_meta["metadatas"]:
            if "source" in m:
                sources.add(m["source"])
        return sorted(sources)

    def clear(self) -> None:
        """Remove all documents from the knowledge base."""
        all_data = self.collection.get(include=[])
        if all_data["ids"]:
            self.collection.delete(ids=all_data["ids"])
        self._manifest.clear()
        self._save_manifest()
        self._rebuild_bm25_index()

    # ─── Manifest ───────────────────────────────────────────────

    def _load_manifest(self):
        """Load document manifest from disk."""
        self._manifest = load_json(self._manifest_path, {})

    def _save_manifest(self):
        """Persist document manifest."""
        os.makedirs(os.path.dirname(self._manifest_path), exist_ok=True)
        atomic_write_json(self._manifest_path, self._manifest)

    # ─── Private helpers ────────────────────────────────────────

    def _split_long_text(self, text: str) -> list[str]:
        """Recursively split text that exceeds chunk_size."""
        if len(text) <= self.chunk_size:
            return [text]

        # Try single newline first
        lines = text.split("\n")
        if len(lines) > 1:
            return self._split_by_boundaries(lines, "\n")

        # Try sentence boundaries (Chinese + English)
        sentences = re.split(r"(?<=[。！？.!?\n])", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > 1:
            return self._split_by_boundaries(sentences, "")

        # Last resort: hard character split
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i : i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        # A trailing fragment <= chunk_overlap carries no new content: the hard
        # split steps by (chunk_size - chunk_overlap), so the previous chunk
        # already covers it entirely. Drop it instead of emitting a redundant
        # tiny chunk that would pollute the index.
        if len(chunks) > 1 and len(chunks[-1]) <= self.chunk_overlap:
            chunks.pop()
        return chunks

    def _split_by_boundaries(self, segments: list[str], joiner: str) -> list[str]:
        """Greedily combine segments into chunks <= chunk_size."""
        chunks = []
        buffer = ""
        for seg in segments:
            candidate = (buffer + joiner + seg).strip() if buffer else seg
            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer)
                buffer = seg
        if buffer:
            chunks.append(buffer)
        return chunks

    def _extract_pdf(self, path: str) -> str:
        """Extract text from a PDF using PyMuPDF (fitz)."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            try:
                import pdfplumber
            except ImportError:
                raise ImportError(
                    "PDF support requires either PyMuPDF or pdfplumber. "
                    "Install with: pip install PyMuPDF"
                )
            with pdfplumber.open(path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n\n".join(pages)

        with fitz.open(path) as doc:
            pages = [page.get_text() for page in doc]
        return "\n\n".join(pages)

    def _extract_json(self, path: str) -> str:
        """Extract text content from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        extracted = []

        def _walk(obj, depth=0):
            if depth > 5:
                return
            if isinstance(obj, dict):
                for key in ("content", "text", "body", "description", "markdown"):
                    if key in obj and isinstance(obj[key], str) and obj[key].strip():
                        extracted.append(obj[key].strip())
                for v in obj.values():
                    _walk(v, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item, depth + 1)
            elif isinstance(obj, str) and len(obj) > 50 and obj.strip():
                extracted.append(obj.strip())

        _walk(data)
        return "\n\n".join(extracted) if extracted else json.dumps(data, ensure_ascii=False, indent=2)
