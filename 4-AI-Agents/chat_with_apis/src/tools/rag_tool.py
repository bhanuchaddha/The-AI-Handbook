from llama_index.core import (
    Settings, 
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    PromptTemplate, 
    Document
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser
import logging
import datetime
from vector_store.qdrant import QdrantStore

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
        
        self.qdrant_store = QdrantStore()
        self.storage_context = self.qdrant_store.storage_context

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
            
            if not self.qdrant_store.is_document_indexed(file_name, doc.text):
                # Add metadata for tracking
                doc.metadata.update({
                    "doc_hash": self.qdrant_store._get_document_hash(doc.text),
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
                        vector_store=self.qdrant_store.vector_store
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
