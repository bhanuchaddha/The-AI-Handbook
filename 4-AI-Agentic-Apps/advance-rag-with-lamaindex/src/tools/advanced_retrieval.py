from llama_index.core import VectorStoreIndex, Settings, PromptTemplate  # Added PromptTemplate
from llama_index.core.tools import ToolMetadata
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.agent import ReActAgent
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.retrievers import RecursiveRetriever, AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import PrevNextNodePostprocessor  # Updated import
from llama_index.core.indices.query.query_transform import HyDEQueryTransform, DecomposeQueryTransform
import logging

class AdvancedRetrieval:
    """
    Advanced RAG implementation with multiple retrieval strategies:
    1. Query transformation and rewriting
    2. Agent-based reasoning
    3. Auto-merging retrieval
    4. Parent-child document relationships
    5. Sub-question decomposition
    """
    
    @staticmethod
    def transform_query(query_str: str) -> str:
        """Apply query transformation techniques"""
        # Convert to instruction format
        transformed_query = f"Find relevant information to answer: {query_str}"
        # Add context request
        transformed_query += "\nProvide specific details and examples if available."
        return transformed_query

    @staticmethod
    def setup_recursive_retrieval(index, similarity_top_k=5):
        """Setup recursive retrieval strategy"""
        try:
            return RecursiveRetriever(
                index=index,
                child_branch_factor=2,
                max_depth=2,
                similarity_top_k=similarity_top_k
            )
        except Exception as e:
            logging.warning(f"Failed to setup recursive retrieval: {e}")
            # Fallback to base retriever
            return index.as_retriever(similarity_top_k=similarity_top_k)

    @staticmethod
    def create_query_pipeline(retriever):
        """Create an advanced query pipeline"""
        try:
            # Use similarity postprocessor with default settings
            return [SimilarityPostprocessor(similarity_cutoff=0.7)]
        except Exception as e:
            logging.warning(f"Failed to create query pipeline: {e}")
            return []

    @staticmethod
    def query_transformation_handler():
        """Create a query transformation handler"""
        # Instead of using QueryTransformation, we'll use a simple function
        return lambda x: AdvancedRetrieval.transform_query(x)

    @staticmethod
    def create_agent_with_tools(index, query_engine):
        """
        Creates a ReAct agent for advanced reasoning over documents.
        The agent can:
        - Search through documents
        - Break down complex queries
        - Provide step-by-step reasoning
        """
        logging.info("Creating ReAct agent with document search capabilities")
        try:
            # Create search tool for the agent
            query_engine_tools = [
                QueryEngineTool(
                    query_engine=query_engine,
                    metadata=ToolMetadata(
                        name="document_search",
                        description="Searches through documents to find relevant information"
                    )
                )
            ]
            
            # Initialize ReAct agent
            agent = ReActAgent.from_tools(
                query_engine_tools,
                verbose=True,
                context="You are an AI assistant helping to find information in documents."
            )
            logging.info("Successfully initialized ReAct agent")
            return agent
        except Exception as e:
            logging.error(f"Failed to create agent: {e}")
            return None

    @staticmethod
    def setup_advanced_retrieval(index, similarity_top_k=5):
        """Sets up advanced hybrid retrieval pipeline"""
        logging.info("Setting up advanced retrieval system")
        
        try:
            # Base vector retriever
            vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
            
            # BM25 retriever for keyword search
            bm25_retriever = BM25Retriever.from_defaults(
                index=index,
                similarity_top_k=similarity_top_k
            )
            
            # Setup HyDE (Hypothetical Document Embeddings) query transform
            hyde_transform = HyDEQueryTransform(include_original=True)
            
            # Setup query decomposition transform
            decompose_transform = DecomposeQueryTransform(
                verbose=True,
                llm=Settings.llm
            )
            
            # Apply transforms to vector retriever
            vector_retriever.query_transform = hyde_transform
            
            # Setup auto-merging retriever
            auto_merge_retriever = AutoMergingRetriever(
                base_retriever=vector_retriever,
                storage_context=index.storage_context,
                verbose=True
            )
            
            # Combine retrievers using fusion (updated implementation)
            retrievers = [vector_retriever, bm25_retriever, auto_merge_retriever]
            fusion_retriever = FusionRetriever(
                retrievers,
                similarity_top_k=similarity_top_k
            )
            
            # Update context processing with correct postprocessor
            node_postprocessors = [
                PrevNextNodePostprocessor(num_prev=2, num_next=2),  # Changed from SentenceWindowNodePostprocessor
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ]
            
            logging.info("Advanced retrieval system ready")
            return fusion_retriever, node_postprocessors
            
        except Exception as e:
            logging.warning(f"Failed to setup advanced retrieval: {e}")
            return index.as_retriever(similarity_top_k=similarity_top_k), None

    @staticmethod
    def get_response_synthesizer():
        """Creates an advanced response synthesizer"""
        template = PromptTemplate(
            "Given the context information, answer the question step by step:\n"
            "Context: {context_str}\n"
            "Question: {query_str}\n"
            "Answer with reasoning:"
        )
        return CompactAndRefine(
            streaming=True,
            verbose=True,
            structured_answer_filtering=True,
            text_qa_template=template
        )

    @staticmethod
    def create_retrieval_query_engine(index, retriever, node_postprocessors):
        """Creates an advanced retrieval query engine"""
        return RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=node_postprocessors,
            response_synthesizer=AdvancedRetrieval.get_response_synthesizer()
        )

    @staticmethod
    def create_advanced_query_pipeline(retrievers, postprocessors=None):
        """
        Creates a query pipeline that:
        1. Uses multiple retrievers in sequence
        2. Applies similarity and time-based filtering
        3. Ranks and filters results
        """
        logging.info("Creating advanced query pipeline")
        
        # Setup default postprocessors if none provided
        if postprocessors is None:
            logging.debug("Using default postprocessors")
            postprocessors = [
                SimilarityPostProcessor(similarity_cutoff=0.7),
                TimeWeightedPostProcessor(time_decay=0.99)
            ]
        
        # Create pipeline
        pipeline = QueryPipeline(
            retrievers=retrievers,
            postprocessors=postprocessors
        )
        logging.info(f"Query pipeline created with {len(retrievers)} retrievers and {len(postprocessors)} postprocessors")
        return pipeline

    @staticmethod
    def setup_sub_question_engine(index):
        """
        Sets up an engine that can:
        1. Break down complex queries into sub-questions
        2. Answer each sub-question separately
        3. Synthesize a complete answer
        """
        logging.info("Initializing sub-question query engine")
        engine = SubQuestionQueryEngine.from_defaults(
            query_engine=index.as_query_engine(),
            verbose=True
        )
        logging.info("Sub-question engine ready")
        return engine
