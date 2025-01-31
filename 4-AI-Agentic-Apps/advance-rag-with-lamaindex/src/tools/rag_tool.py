from llama_index.core import (
    Settings, 
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    PromptTemplate, 
    Document,
    Response
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
import logging
import datetime
from vector_store.qdrant import QdrantStore
from typing import Dict
from llama_index.core.base.llms.types import ChatMessage
from .advanced_retrieval import AdvancedRetrieval

class RagTool:
    def __init__(self):
        self.llm = Ollama(model="deepseek-r1:8b", request_timeout=120.0)
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
            "Given the context information and query, think through these steps:\n"
            "1. Analyze the relevant context\n"
            "2. Identify key information\n"
            "3. Formulate a clear answer\n\n"
            "Think step by step before answering. If no relevant information is found, say 'I cannot find relevant information to answer your question.'\n"
            "Query: {query_str}\n"
            "Thought Process and Answer: "
        ))
        
        self.qdrant_store = QdrantStore(collection_name="document_store")
        self.storage_context = self.qdrant_store.storage_context

        # Remove debug handler, we'll use direct retrieval instead
        self.retrieved_nodes = []
        
        # Initialize index from existing documents
        self.initialize_from_qdrant()
        self.advanced_retrieval = AdvancedRetrieval()
        self.agent = None
        self.sub_question_engine = None

    def initialize_from_qdrant(self):
        """Initialize the index from existing Qdrant store"""
        try:
            collection_info = self.qdrant_store.client.get_collection(self.qdrant_store.collection_name)
            if collection_info.points_count > 0:
                logging.info(f"Found existing collection with {collection_info.points_count} documents")
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.qdrant_store.vector_store,
                    show_progress=True
                )
                logging.info("Successfully loaded existing documents from Qdrant")
            else:
                logging.info("No existing documents found in Qdrant")
        except Exception as e:
            logging.error(f"Error loading existing documents: {e}")
            self.index = None

    def process_document(self, temp_dir):
        reader = DoclingReader()
        loader = SimpleDirectoryReader(
            input_dir=temp_dir,
            file_extractor={
                "*": reader  # Use DoclingReader for all file types
            },
        )
        
        try:
            docs = loader.load_data()
        except Exception as e:
            logging.error(f"Error loading document: {str(e)}")
            raise Exception(f"Failed to process document: {str(e)}")
        
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
        query_engine = self.index.as_query_engine(
            streaming=True,
            verbose=True,  # Add verbose mode
            similarity_top_k=10,  # Show top 2 similar chunks
        )
        query_engine.update_prompts({"response_synthesizer:text_qa_template": self.qa_prompt_tmpl})
        
        return query_engine

    def initialize_advanced_features(self):
        """Initialize advanced RAG features"""
        if self.index is None:
            return

        # Setup advanced retrievers
        auto_merging_retriever, parent_child_retriever = self.advanced_retrieval.setup_advanced_retrieval(self.index)
        
        # Setup query engine with advanced features
        query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact",
            structured_answer_filtering=True
        )
        
        # Initialize agent
        self.agent = self.advanced_retrieval.create_agent_with_tools(self.index, query_engine)
        
        # Initialize sub-question engine
        self.sub_question_engine = self.advanced_retrieval.setup_sub_question_engine(self.index)

    def get_query_engine(self):
        """Get initialized query engine with all advanced features"""
        if self.index is None:
            self.initialize_from_qdrant()
            if self.index is None:
                raise Exception("No documents have been indexed yet")
        
        try:
            # Get advanced retriever and postprocessors
            retriever, node_postprocessors = self.advanced_retrieval.setup_advanced_retrieval(
                self.index, 
                similarity_top_k=5
            )
            
            # Create advanced query engine
            query_engine = self.advanced_retrieval.create_retrieval_query_engine(
                self.index,
                retriever,
                node_postprocessors
            )
            
            query_engine.update_prompts({"response_synthesizer:text_qa_template": self.qa_prompt_tmpl})
            return query_engine
            
        except Exception as e:
            logging.error(f"Error setting up query engine: {e}")
            return self.index.as_query_engine(
                similarity_top_k=5,
                response_mode="compact"
            )

    def query_with_debug(self, query_str: str) -> Response:
        """
        Multi-stage query processing with fallback strategies:
        1. Agent-based reasoning
        2. Sub-question decomposition
        3. Standard retrieval
        """
        logging.info(f"\n{'='*50}\nProcessing Query: {query_str}\n{'='*50}")
        
        if not self.index:
            logging.error("No index available for query")
            raise Exception("No documents indexed yet")
        
        try:
            # Stage 1: Query Transformation
            logging.info("\n🔄 Stage 1: Query Transformation")
            transformed_query = self.advanced_retrieval.transform_query(query_str)
            logging.info(f"Original query: {query_str}")
            logging.info(f"Transformed query: {transformed_query}")
            
            # Stage 2: Agent-based Approach
            logging.info("\n🤖 Stage 2: Attempting Agent-based Processing")
            if self.agent:
                try:
                    logging.info("Executing agent-based reasoning...")
                    agent_response = self.agent.chat(transformed_query)
                    if agent_response and agent_response.response:
                        logging.info("✅ Agent successfully processed query")
                        logging.debug(f"Agent response: {agent_response.response}")
                        return agent_response.response
                except Exception as e:
                    logging.warning(f"❌ Agent-based query failed: {e}")
                    logging.info("Falling back to sub-question decomposition...")
            
            # Stage 3: Sub-question Decomposition
            logging.info("\n🔍 Stage 3: Attempting Sub-question Decomposition")
            if self.sub_question_engine and len(query_str.split()) > 15:
                try:
                    logging.info("Query is complex, breaking down into sub-questions...")
                    sub_question_response = self.sub_question_engine.query(transformed_query)
                    if sub_question_response:
                        logging.info("✅ Successfully processed with sub-questions")
                        logging.debug(f"Sub-question response: {sub_question_response}")
                        return sub_question_response
                except Exception as e:
                    logging.warning(f"❌ Sub-question processing failed: {e}")
                    logging.info("Falling back to standard retrieval...")
            
            # Stage 4: Standard Retrieval
            logging.info("\n📚 Stage 4: Standard Retrieval Processing")
            logging.info("Initializing advanced query engine...")
            query_engine = self.get_query_engine()
            
            logging.info("Executing final query...")
            response = query_engine.query(transformed_query)
            logging.info("✅ Query processing complete")
            
            # Log retrieval statistics
            if hasattr(response, 'metadata'):
                logging.info("\n📊 Retrieval Statistics:")
                logging.info(f"Number of source nodes: {len(self.retrieved_nodes)}")
                logging.info(f"Response length: {len(str(response))}")
            
            return response
            
        except Exception as e:
            logging.error(f"❌ Error during retrieval: {e}")
            raise

    def get_query_debug_info(self):
        """Get debug info about the last query"""
        if not self.retrieved_nodes:
            return None
            
        return {
            "nodes_by_relevance": self.retrieved_nodes
        }
