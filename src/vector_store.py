# src/vector_store.py
import uuid
import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import VECTOR_DB_PATH, COLLECTION_NAME
from langchain_core.documents import Document

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VectorStore:
    """
    A simple wrapper around Qdrant to manage vector embeddings and documents.
    Single responsibility: handle vector DB operations (insert, search, stats).
    """

    def __init__(self, embedding_model):
        """
        Args:
            embedding_model: An embedding model object with .embed(list[str]) method.
        """
        self.client = QdrantClient(path=VECTOR_DB_PATH)
        self.collection_name = COLLECTION_NAME
        self.embedding_model = embedding_model
        self.vector_size: Optional[int] = None

    def ensure_or_create_collection(self, vector_size: int) -> None:
        """
        Ensure the collection exists, or create it if it does not.
        """
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} already exists.")
        except Exception:
            logger.info(f"Creating collection {self.collection_name} with vector_size={vector_size}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def add_documents(self, docs: List[Document]) -> bool:
        """
        Insert documents and their embeddings into the collection.

        Args:
            docs: A list of LangChain Document objects with .page_content and .metadata
        """
        if not docs:
            logger.warning("No documents provided for insertion.")
            return False

        texts = [doc.page_content for doc in docs]
        embeddings = list(self.embedding_model.embed(texts))

        if not embeddings:
            logger.error("Failed to generate embeddings.")
            return False

        if self.vector_size is None:
            self.vector_size = len(embeddings[0])
            self.ensure_or_create_collection(self.vector_size)

        points = []
        for doc, emb in zip(docs, embeddings):
            payload = {
                "text": doc.page_content,
                **doc.metadata,  # keep all metadata: source, page, region, etc.
            }
            point = PointStruct(
                id=str(uuid.uuid4()),  # unique ID
                vector=emb,
                payload=payload,
            )
            points.append(point)

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Inserted {len(points)} documents into {self.collection_name}")
        return True

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        region: Optional[str] = None,
        comuna: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.

        Args:
            query: Input query string.
            k: Number of results to return.
            region: Optional region filter.
            comuna: Optional comuna filter.

        Returns:
            List of dicts with 'page_content' and 'metadata'
        """
        query_emb = list(self.embedding_model.embed([query]))[0]

        must_filters = []
        if region:
            must_filters.append({"key": "region", "match": {"value": region}})
        if comuna:
            must_filters.append({"key": "comuna", "match": {"value": comuna}})

        query_filter = {"must": must_filters} if must_filters else None

        result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_emb,
            limit=k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

        docs = []
        for hit in result:
            docs.append(
                {
                    "page_content": hit.payload.get("text", ""),
                    "metadata": {**hit.payload, "score": hit.score},
                }
            )

        logger.info(f"Found {len(docs)} relevant documents.")
        return docs

    def get_stats(self) -> Dict[str, Any]:
        """
        Return basic statistics about the collection.
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "exists": True,
                "points_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.value,
            }
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def delete_collection(self) -> bool:
        """
        Completely delete the collection.
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} deleted.")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            return False
