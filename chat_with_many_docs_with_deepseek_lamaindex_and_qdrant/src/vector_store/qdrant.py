import qdrant_client
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
import logging
import hashlib

class QdrantStore:
    def __init__(self, collection_name: str = "document_store", embed_dimension: int = 1024):
        self.collection_name = collection_name
        self.embed_dimension = embed_dimension
        self.qdrant_client = qdrant_client.QdrantClient(
            host="localhost",
            port=6333
        )
        self._ensure_collection_exists()
        
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
        )
        
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def _ensure_collection_exists(self):
        """Initialize Qdrant collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections().collections
            exists = any(col.name == self.collection_name for col in collections)
            if not exists:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embed_dimension,
                        distance=Distance.COSINE
                    )
                )
                logging.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
            logging.error(f"Error creating collection: {e}")
            raise

    def _get_document_hash(self, content: str) -> str:
        """Generate unique hash for document content"""
        return hashlib.md5(content.encode()).hexdigest()

    def is_document_indexed(self, file_name: str, content: str) -> bool:
        """Check if document is already indexed in Qdrant"""
        doc_hash = self._get_document_hash(content)
        try:
            points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="doc_hash",
                            match={"value": doc_hash}
                        )
                    ]
                ),
                limit=1,
                with_payload=True
            )[0]
            
            exists = len(points) > 0
            if exists:
                logging.info(f"Document already indexed: {file_name} (hash: {doc_hash})")
            return exists
        except Exception as e:
            logging.error(f"Error checking document index: {e}")
            return False

    @property
    def client(self):
        return self.qdrant_client
