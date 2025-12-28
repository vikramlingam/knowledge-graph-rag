"""
RAG (Retrieval Augmented Generation) module using local SentenceTransformer embeddings.
Uses all-MiniLM-L6-v2 for lightweight, CPU-efficient embeddings.
"""

import faiss
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

# Model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
MODELS_DIR = Path(__file__).parent.parent / "models" / "embeddings"


class LocalEmbeddings:
    """Local embedding model using SentenceTransformers."""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._load_model()
    
    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Set cache directory to our models folder
            cache_dir = MODELS_DIR
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
            self._model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                cache_folder=str(cache_dir),
                device="cpu"
            )
            logger.info("Embedding model loaded successfully.")
            
        except ImportError as e:
            logger.error(f"sentence-transformers not installed: {e}")
            raise RuntimeError("Install sentence-transformers") from e
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if not texts:
            return np.array([]).reshape(0, EMBEDDING_DIMENSIONS)
        
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.astype("float32")
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to embedding."""
        return self.encode([text])[0]


# Global embeddings instance
_embeddings = None

def get_embeddings() -> LocalEmbeddings:
    """Get or create the global embeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = LocalEmbeddings()
    return _embeddings


def build_vector_store(chunks: List[dict]) -> Tuple[Any, List[dict]]:
    """
    Creates embeddings for text chunks and builds a FAISS index.
    
    Args:
        chunks: List of dicts with 'text', 'source', 'page' keys
        
    Returns:
        Tuple of (FAISS index, chunks list) or (None, []) on error
    """
    if not chunks:
        return None, []

    embeddings = get_embeddings()
    texts_to_embed = [chunk['text'] for chunk in chunks]
    
    try:
        logger.info(f"Creating embeddings for {len(texts_to_embed)} chunks...")
        all_embeddings = embeddings.encode(texts_to_embed)
        
        if len(all_embeddings) == 0:
            return None, []
        
        # Build FAISS index
        index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)  # Inner product for normalized vectors
        index.add(all_embeddings)
        
        logger.info(f"Built FAISS index with {index.ntotal} vectors")
        return index, chunks
        
    except Exception as e:
        logger.error(f"Error building vector store: {e}")
        return None, []


def retrieve(query: str, vector_store: Tuple[Any, List[dict]], k: int = 7) -> List[dict]:
    """
    Retrieves relevant chunks from the vector store based on the query.
    
    Args:
        query: The search query
        vector_store: Tuple of (FAISS index, chunks list)
        k: Number of results to return
        
    Returns:
        List of relevant chunk dicts
    """
    index, chunks = vector_store
    if index is None or not chunks:
        return []

    embeddings = get_embeddings()
    
    try:
        query_embedding = embeddings.encode_single(query).reshape(1, -1)
        distances, indices = index.search(query_embedding, k)
        
        relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        return []
