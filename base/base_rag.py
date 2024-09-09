from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Redis as LangchainRedis
from langchain.text_splitter import RecursiveCharacterTextSplitter
from redisvl.extensions.llmcache.semantic import SemanticCache
import os


class BaseRag:
    def __init__(self, config_rag) -> None:
        self.config_rag = config_rag
        self.embeddings = self._get_embedding_model()
        self.vector_store = self._configure_vector_store()
        self.cache = self._configure_cache()

    # def _get_vector_store(self, embeddings):
    #     """Returns the Chroma Vector Store"""
    #     vector_store = Chroma(
    #         collection_name=self.config_rag['collection_name'],
    #         embedding_function=embeddings,
    #         persist_directory=self.config_rag['embedding_store_dir']
    #     )
    #     return vector_store

    def _get_embedding_model(self):
        """Returns the Embedding Model."""
        embeddings = OllamaEmbeddings(
            model=self.config_rag['embedding_model'],
        )
        return embeddings

    def _configure_vector_store(self):
        """
        Configure the Redis vector store by loading, splitting, and embedding documents from the given path.
        Args:
        Returns:
            retriever: A retriever object for querying the vector store.
        """
        stored_pdf_dir: str = self.config_rag['pdf_dir']

        docs: list = []

        # animation_loader: Loader = Loader(
        #     desc="Loading pdf", end="pdf loaded ✅", timeout=0.05
        # ).start()
        # Load PDF documents from the specified directory
        for folders in os.listdir(stored_pdf_dir):
            path_of_individual_folders: str = os.path.join(stored_pdf_dir, folders)
            for file in os.listdir(path_of_individual_folders):
                if file.endswith(".pdf"):  # Process only PDF files
                    pdf_loader: PyPDFLoader = PyPDFLoader(
                        os.path.join(path_of_individual_folders, file)
                    )  # Create a loader for the PDF

                    docs.extend(pdf_loader.load())  # Load the PDF content
        # Split the loaded documents into chunks
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )

        splits: list = text_splitter.split_documents(docs)  # Split documents into chunks

        # Create and configure the Redis vector store
        vectordb: LangchainRedis = LangchainRedis.from_documents(
            documents=splits,
            embedding=self.embeddings,
            redis_url=self.config_rag['redis_url'],
            index_name=self.config_rag['index_name']
        )
        # animation_loader.stop()
        # Return a retriever for querying the vector store
        return vectordb.as_retriever()

    def _configure_cache(self) -> SemanticCache:
        """
        Configure the LLM cache using Redis for semantic caching.
        Returns:
            cache: A SemanticCache object for caching LLM responses.
        """

        return SemanticCache(
            redis_url=self.config_rag['redis_url'],
            distance_threshold=self.config_rag['llm_cache_threshold'],  # Semantic similarity threshold
            name="llmcache",  # Name for the cache
        )


    def ask_question(self) -> str:
        """Take user prompt"""
        raise NotImplementedError

    def check_cache(self, user_prompt: str):
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
