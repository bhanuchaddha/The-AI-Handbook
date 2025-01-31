from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser

class RagTool:
    def __init__(self):
        self.llm = Ollama(model="codellama:7b", request_timeout=120.0)
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

    def process_document(self, temp_dir):
        reader = DoclingReader()
        loader = SimpleDirectoryReader(
            input_dir=temp_dir,
            file_extractor={".pdf": reader},
        )
        docs = loader.load_data()
        
        node_parser = MarkdownNodeParser()
        print("Indexing begins...")
        index = VectorStoreIndex.from_documents(documents=docs, transformations=[node_parser], show_progress=True)
        print("Indexing complete!")
        query_engine = index.as_query_engine(streaming=True)
        
        qa_prompt_tmpl_str = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information above I want you to think step by step to answer the query in a highly precise and crisp manner focused on the final answer, incase case you don't know the answer say 'I don't know!'.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
        
        return query_engine
