"""Learned reranker for improving relevance of hybrid search results.

Applies cross-encoder models to rerank top-K results from hybrid search,
providing more nuanced relevance scoring for technical document retrieval.

Supports:
- Multiple cross-encoder backends (sentence-transformers)
- Caching of reranker scores
- Batch processing for efficiency
- Configurable top-K reranking
- Integration with existing hybrid search pipeline

Architecture:
    1. Hybrid search returns top-K results with fusion scores
    2. Reranker takes (query, top-K documents) pairs
    3. Cross-encoder scores each pair independently
    4. Results reordered by reranker score

Models (popular for technical docs):
    - BAAI/bge-reranker-base (5.3K NDCG@10, 278M params)
    - BAAI/bge-reranker-large (6.2K NDCG@10, 919M params)
    - cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (lightweight)
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RankerResult:
    """Single reranked result with scores."""

    doc_id: str
    reranker_score: float
    hybrid_score: Optional[float] = None
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "doc_id": self.doc_id,
            "reranker_score": self.reranker_score,
            "hybrid_score": self.hybrid_score,
            "rank": self.rank,
        }


class CrossEncoderReranker:
    """Reranks search results using cross-encoder models.

    Cross-encoders score (query, document) pairs jointly, capturing
    semantic alignment better than independent embeddings. Ideal for
    technical document retrieval with nuanced relevance signals.

    Features:
    - Lazy model loading (loads on first rerank call)
    - Result caching with optional disk persistence
    - Batch inference for efficiency
    - Configurable device (CPU/CUDA/MPS)
    - Logging and metrics
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
        enable_cache: bool = True,
        cache_dir: Optional[str] = None,
        strict_offline: bool = False,
    ):
        """Initialise cross-encoder reranker.

        Args:
            model_name: HuggingFace model identifier (default: BAAI/bge-reranker-base)
            device: Device for inference ("cpu", "cuda", "mps"); auto-detects if available
            batch_size: Batch size for inference (32 recommended)
            max_length: Max sequence length for tokeniser (512 typical)
            enable_cache: Cache reranker scores to memory and optionally disk
            cache_dir: Directory for persistent score cache (~/.cache/rag_reranker default)
            strict_offline: If True, disallow all Hugging Face network access and require
            model/tokeniser files to already exist in local cache.

            TODO: Evaluate BAAI/bge-reranker-v2-m3, can take longer to initialise, but can be faster and more accurate at inference time.
            TODO: Add support for other cross-encoder reranker libraries.

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.enable_cache = enable_cache
        self.strict_offline = strict_offline

        # Set cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / ".cache" / "rag_reranker"

        self._model = None
        self._model_loaded = False
        self._cache: Dict[str, float] = {}

        # Load cache if available
        if self.enable_cache:
            self._load_cache()

    def _load_model(self) -> None:
        """Lazy-load cross-encoder model on first use."""
        if self._model_loaded:
            return

        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "sentence-transformers required for reranking. "
                "Install with: pip install sentence-transformers"
            )

        try:
            logger.info(f"Loading cross-encoder: {self.model_name}")
            if self.strict_offline:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

                hf_home = Path(
                    os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
                )
                transformers_cache = Path(
                    os.environ.get(
                        "TRANSFORMERS_CACHE",
                        str(hf_home / "hub"),
                    )
                )
                sentence_transformers_cache = Path(
                    os.environ.get(
                        "SENTENCE_TRANSFORMERS_HOME",
                        str(Path.home() / ".cache" / "torch" / "sentence_transformers"),
                    )
                )

                logger.warning(
                    "Reranker strict offline mode is enabled. Network downloads are disabled. "
                    f"Ensure model '{self.model_name}' is already present in local cache paths: "
                    f"HF_HOME={hf_home}, TRANSFORMERS_CACHE={transformers_cache}, "
                    f"SENTENCE_TRANSFORMERS_HOME={sentence_transformers_cache}"
                )

            cross_encoder_kwargs: Dict[str, Any] = {
                "device": self.device,
                "max_length": self.max_length,
            }

            if self.strict_offline:
                cross_encoder_kwargs["local_files_only"] = True

            try:
                self._model = CrossEncoder(self.model_name, **cross_encoder_kwargs)
            except TypeError as exc:
                # Some sentence-transformers versions do not accept local_files_only
                # directly on CrossEncoder constructor; fallback to env-only offline mode.
                if self.strict_offline and "local_files_only" in str(exc):
                    cross_encoder_kwargs.pop("local_files_only", None)
                    self._model = CrossEncoder(self.model_name, **cross_encoder_kwargs)
                else:
                    raise

            self._model_loaded = True
            logger.info(f"✓ Loaded cross-encoder on {self.device}")

            if self.strict_offline:
                logger.info("Reranker strict offline mode enabled (HF_HUB_OFFLINE=1)")

            if "bge-reranker" in self.model_name.lower():
                logger.info(
                    "If Hugging Face shows 'roberta.embeddings.position_ids | UNEXPECTED' "
                    "during model load, this is expected for this reranker family and is safe "
                    "to ignore when inference succeeds."
                )
        except Exception as e:
            logger.error(f"Failed to load cross-encoder {self.model_name}: {e}")
            raise

    def _get_cache_key(self, query: str, doc_id: str) -> str:
        """Generate MD5 cache key for query-document pair."""
        combined = f"{query}||{doc_id}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _load_cache(self) -> None:
        """Load score cache from disk if available."""
        try:
            cache_file = self.cache_dir / "scores.jsonl"
            if cache_file.exists():
                with open(cache_file, "r") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            self._cache[entry["key"]] = entry["score"]
                        except (json.JSONDecodeError, KeyError):
                            pass
                logger.debug(f"Loaded {len(self._cache)} cached reranker scores")
        except Exception as e:
            logger.debug(f"Could not load reranker cache: {e}")

    def _save_cache_entry(self, key: str, score: float) -> None:
        """Append single score entry to cache file."""
        if not self.enable_cache:
            return

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / "scores.jsonl"

            with open(cache_file, "a") as f:
                json.dump({"key": key, "score": float(score)}, f)
                f.write("\n")
        except Exception as e:
            logger.debug(f"Could not save reranker cache entry: {e}")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[RankerResult]:
        """Rerank documents for a query using cross-encoder.

        Args:
            query: Search query
            documents: List of documents with schema:
                {
                    "doc_id": str (required),
                    "text": str (required, document content),
                    "hybrid_score": float (optional, original fusion score)
                }
            top_k: Return only top K results (None = all)

        Returns:
            List of RankerResult sorted by reranker_score (descending)

        Raises:
            ValueError: If documents lack required fields
        """
        if not documents:
            logger.debug("No documents to rerank")
            return []

        # Load model if needed
        if not self._model_loaded:
            self._load_model()

        # Validate and prepare documents
        pairs = []
        doc_info = []

        for doc in documents:
            doc_id = doc.get("doc_id", "").strip()
            text = doc.get("text", "").strip()

            if not doc_id:
                logger.warning("Skipping document missing doc_id")
                continue
            if not text:
                logger.warning(f"Skipping document {doc_id} with no text")
                continue

            pairs.append([query, text])
            doc_info.append(
                {
                    "doc_id": doc_id,
                    "hybrid_score": doc.get("hybrid_score"),
                }
            )

        if not pairs:
            logger.warning("No valid documents to rerank")
            return []

        logger.debug(f"Reranking {len(pairs)} documents")

        # Score in batches with caching
        scores = []

        for i in range(0, len(pairs), self.batch_size):
            batch_end = min(i + self.batch_size, len(pairs))
            batch = pairs[i:batch_end]
            batch_info = doc_info[i:batch_end]

            batch_scores = []
            to_score_indices = []

            # Check cache
            for j, (pair, info) in enumerate(zip(batch, batch_info)):
                cache_key = self._get_cache_key(pair[0], info["doc_id"])

                if cache_key in self._cache:
                    batch_scores.append(self._cache[cache_key])
                else:
                    to_score_indices.append((j, cache_key, info["doc_id"]))

            # Score uncached pairs
            if to_score_indices:
                uncached_pairs = [batch[j] for j, _, _ in to_score_indices]
                uncached_scores = self._model.predict(uncached_pairs)

                for (j, cache_key, doc_id), score in zip(to_score_indices, uncached_scores):
                    batch_scores.insert(j, float(score))
                    self._cache[cache_key] = float(score)
                    self._save_cache_entry(cache_key, float(score))

            scores.extend(batch_scores)

        # Build results
        results = []
        for info, score in zip(doc_info, scores):
            results.append(
                RankerResult(
                    doc_id=info["doc_id"],
                    reranker_score=float(score),
                    hybrid_score=info["hybrid_score"],
                )
            )

        # Sort by reranker score (descending)
        results.sort(key=lambda x: x.reranker_score, reverse=True)

        # Assign final ranks
        for rank, result in enumerate(results, 1):
            result.rank = rank

        # Truncate to top_k if specified
        if top_k:
            results = results[:top_k]

        return results


class RerankerConfig:
    """Configuration for reranker in RAG pipeline."""

    def __init__(
        self,
        enable_reranking: bool = True,
        model_name: str = "BAAI/bge-reranker-base",
        rerank_top_k: int = 50,  # Rerank top 50 from hybrid search
        final_top_k: int = 10,  # Return top 10 after reranking
        device: str = "cpu",
        batch_size: int = 32,
        enable_cache: bool = True,
        cache_dir: Optional[str] = None,
        strict_offline: bool = False,
    ):
        """Initialise reranker configuration.

        Args:
            enable_reranking: Enable reranker in RAG pipeline
            model_name: HuggingFace cross-encoder model
            rerank_top_k: Rerank top K candidates from hybrid search (cost saving)
            final_top_k: Return top K results after reranking
            device: Inference device (cpu, cuda, mps)
            batch_size: Batch size for inference
            enable_cache: Cache reranker scores
            cache_dir: Cache directory path
            strict_offline: Disable Hugging Face network access and use local files only
        """
        self.enable_reranking = enable_reranking
        self.model_name = model_name
        self.rerank_top_k = rerank_top_k
        self.final_top_k = final_top_k
        self.device = device
        self.batch_size = batch_size
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        self.strict_offline = strict_offline


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    config: RerankerConfig,
) -> List[RankerResult]:
    """Convenience function to rerank search results.

    Integrates reranking into RAG pipeline with configuration.

    Args:
        query: Search query
        results: Results from hybrid search pipeline
                 Expected keys: doc_id, text, hybrid_score
        config: RerankerConfig instance

    Returns:
        List of RankerResult sorted by relevance

    Example:
        >>> config = RerankerConfig(enable_reranking=True)
        >>> reranked = rerank_results(query, hybrid_results, config)
        >>> top_10 = [r.doc_id for r in reranked[:10]]
    """
    if not config.enable_reranking or not results:
        # Return results as-is if reranking disabled
        logger.debug("Reranking disabled, returning results unchanged")
        return [
            RankerResult(
                doc_id=r.get("doc_id", ""),
                reranker_score=r.get("hybrid_score", 0.0),
                hybrid_score=r.get("hybrid_score"),
            )
            for r in results
        ]

    reranker = CrossEncoderReranker(
        model_name=config.model_name,
        device=config.device,
        batch_size=config.batch_size,
        enable_cache=config.enable_cache,
        cache_dir=config.cache_dir,
        strict_offline=config.strict_offline,
    )

    # Rerank top K from hybrid search (cost optimisation)
    top_k_for_reranking = min(config.rerank_top_k, len(results))
    to_rerank = results[:top_k_for_reranking]

    logger.info(
        f"Reranking {len(to_rerank)} results (top {top_k_for_reranking} from hybrid search)"
    )

    reranked = reranker.rerank(query, to_rerank, top_k=config.final_top_k)

    logger.info(f"Reranking complete: {len(to_rerank)} candidates → {len(reranked)} final results")

    return reranked
