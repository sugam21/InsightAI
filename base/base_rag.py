from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class BaseRag:
    def __init__(self, config_llm) -> None:
        self.config_llm = config_llm
        self.embeddings = self._get_embedding_model()
        self.vector_store = self._get_vector_store(self.embeddings)

    def _get_vector_store(self, embeddings):
        """Returns the Chroma Vector Store"""
        vector_store = Chroma(
            collection_name=self.config_llm['collection_name'],
            embedding_function=embeddings,
            persist_directory=self.config_llm['embedding_store_dir']
        )
        return vector_store

    def _get_embedding_model(self):
        """Returns the Embedding Model."""
        embeddings = OllamaEmbeddings(
            model=self.config_llm['embedding_model'],
        )
        return embeddings

    def get_prompt(self) -> str:
        """Take user prompt"""
        raise NotImplementedError

    def check_cache(self, prompt: str):
        """Checks Reddis cache and returns the response. If miss then calls get_response function"""
        raise NotImplementedError

    def get_response(self, prompt: str):
        """Use Vector db to fetch the response"""
        raise NotImplementedError

    def format_response(self, prompt: str):
        """Format the response with LLM."""
        raise NotImplementedError

    def save_cache(self, prompt: str, response: str):
        """Save the prompt and response to the cache."""
        raise NotImplementedError
