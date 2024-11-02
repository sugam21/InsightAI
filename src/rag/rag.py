from pathlib import Path

import chromadb
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from loguru import logger

from src import Config

RAG_CONFIG_PATH: str = Path("rag_config.json").resolve()

class_name_mapping_dict = {
    "c0": "Alienware alpha or Alienware steam machine",
    "c1": "XPS 27 7760",
    "c2": "Alienware 13 R3",
    "c3": "Dell Alienware m16 R1",
    "c4": "Alienware m17 R4",
    "c5": "Alienware x17 R2",
    "c6": "Chromebook 11 3180",
    "c7": "Dell G15 5510",
    "c8": "ASUS ROG Strix SCAR 17 (2023)",
    "c9": "ROG Zephyrus G16 (2024) GU605",
    "c10": "Dell XPS 13 9370",
    "c11": "Dell XPS 14 9440",
    "c12": "Dell XPS 15 9500",
    "c13": "Dell XPS 16 9640",
    "c14": "XPS 17 9730",
    "c15": "Dell Alienware m16 R2",
    "c16": "Alienware x14 R2",
}


class RagPipeline:
    def __init__(self, rag_config_path) -> None:
        self.config = Config.from_json(rag_config_path)
        self._setup_path()
        self._setup_models()
        self._setup_vector_store()

    def _setup_path(self):
        """Fixes the relative path problem. So basically in order to go to the required dir, it goes 3 parents up
        from the current file and append the data directory there."""
        self.config.data["data_dir"] = Path(self.config.data["data_dir"]).resolve()
        logger.debug(f"Data Directory {self.config.data['data_dir']}")
        self.config.data["persist_directory"] = Path(
            self.config.data["persist_directory"]
        ).resolve()
        logger.debug(f"Persist directory {self.config.data["persist_directory"]}")

    def _setup_models(self):
        """Set's up embedding models and llm models"""
        self.embeddings = OllamaEmbeddings(
            model=self.config.train["model"]["embedding_model"],
        )
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = ChatOllama(
            model=self.config.train["model"]["llm_model"],
            num_thread=8,
            verbose=True,
            num_gpu=1,
            callbacks=callback_manager,
        )

    def _setup_vector_store(self):
        print(
            "Setup Vector Store Path",
            Path(self.config.data["persist_directory"]).resolve(),
        )
        persistent_client = chromadb.PersistentClient(
            path=str(self.config.data["persist_directory"])
        )
        self.vector_store = Chroma(
            client=persistent_client,
            collection_name=self.config.train["model"]["collection_name"],
            embedding_function=self.embeddings,
        )

    @staticmethod
    def format_docs(docs):
        "\n\n".join(doc.page_content for doc in docs)

    def run(self, query: str, image_class: str):
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.config.train["output_document_from_vector_store"],
                "filter": {"category": image_class},
            },
        )
        logger.debug(f"Image class {image_class}")
        logger.debug(f"User query {query}")

        retrived_query = retriever.invoke(query)
        formatted_docs = "\n\n".join(doc.page_content for doc in retrived_query)
        result = self.llm.invoke(
            f"""You are an expert assistant for question answering tasks.
         Use the following context information to answer the question. If there is something repeating
        in the conetne do not repeat that into the response.
         If the context does not contain answer, just say that you don't know.\n\n
         Question: {query}
         Context:{formatted_docs}"""
        )
        return result


if __name__ == "__main__":
    rp = RagPipeline(rag_config_path=RAG_CONFIG_PATH)
    query = "Show me step by step on setting up the computer ?"
    image_class = (class_name_mapping_dict["c14"]).strip()
    print(query)
    print(image_class)
    while True:
        result = rp.run(query, image_class)
        print(result.content)
        break
