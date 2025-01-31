from llama_index.core import (
    Settings, 
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    PromptTemplate, 
    StorageContext,
    Document,
    ServiceContext
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import hashlib
import logging
import datetime
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

class RagTool:
    def __init__(self):
        self.llm = Ollama(model="codellama:7b", request_timeout=120.0)
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.node_parser = MarkdownNodeParser()
        self.index = None
        self.qa_prompt_tmpl = PromptTemplate((
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information above I want you to think step by step to answer the query in a highly precise and crisp manner focused on the final answer, incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
        ))
        self.collection_name = "pdf_documents"
        
        # Configure Qdrant with explicit vector parameters
        self.qdrant_client = qdrant_client.QdrantClient(
            host="localhost",
            port=6333
        )
        
        # Get embedding dimension from the model
        self.embed_dimension = 1024  # BGE-large dimension
        
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
            # Use payload API for more reliable metadata search
            points = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="doc_hash",  # Remove metadata. prefix
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

    def process_document(self, temp_dir):
        reader = DoclingReader()
        loader = SimpleDirectoryReader(
            input_dir=temp_dir,
            file_extractor={".pdf": reader},
        )
        docs = loader.load_data()
        
        # Filter out already indexed documents
        new_docs = []
        for doc in docs:
            file_name = doc.metadata.get('file_name', 'unknown')
            doc_hash = self._get_document_hash(doc.text)
            
            if not self.is_document_indexed(file_name, doc.text):
                # Add metadata for tracking
                doc.metadata.update({
                    "doc_hash": doc_hash,
                    "file_name": file_name,
                    "indexed_at": datetime.datetime.now().isoformat(),
                })
                new_docs.append(doc)
                logging.info(f"Adding new document to index: {file_name}")
            else:
                logging.info(f"Skipping already indexed document: {file_name}")
        
        if not new_docs:
            logging.info("No new documents to index")
            if self.index is None:
                try:
                    self.index = VectorStoreIndex.from_vector_store(
                        vector_store=self.vector_store
                    )
                    logging.info("Loaded existing index from vector store")
                except Exception as e:
                    logging.error(f"Error loading existing index: {e}")
                    self.index = None
                    
            return self.index.as_query_engine(streaming=True)

        logging.info(f"Indexing {len(new_docs)} new documents...")
        
        if self.index is None:
            self.index = VectorStoreIndex.from_documents(
                documents=new_docs, 
                storage_context=self.storage_context,
                show_progress=True
            )
        else:
            # Create nodes and insert them directly
            nodes = Settings.node_parser.get_nodes_from_documents(new_docs)
            self.index.insert_nodes(nodes)
            
        logging.info("Indexing complete!")
        query_engine = self.index.as_query_engine(streaming=True)
        query_engine.update_prompts({"response_synthesizer:text_qa_template": self.qa_prompt_tmpl})
        
        return query_engine
