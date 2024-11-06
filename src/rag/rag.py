import os
from pathlib import Path

import chromadb
import dotenv
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from loguru import logger

from src import Config

RAG_CONFIG_PATH: str = Path("rag_config.json").resolve()
dotenv.load_dotenv()

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
        self.embeddings = AzureOpenAIEmbeddings(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            model="text-embedding-3-large",
            api_version="2023-05-15",
            dimensions=1024,
        )
        # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        # self.llm = ChatOllama(
        #     model=self.config.train["model"]["llm_model"],
        #     num_thread=8,
        #     verbose=True,
        #     num_gpu=1,
        #     callbacks=callback_manager,
        # )

        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version="2024-08-01-preview",
            model="gpt-4",
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            openai_api_type="azure",
            temperature=0.3,
            streaming=True,
        )

    def _setup_vector_store(self):
        logger.info(
            f"Vector Store Path {Path(self.config.data["persist_directory"]).resolve()}"
        )
        persistent_client = chromadb.PersistentClient(
            path=str(self.config.data["persist_directory"])
        )
        self.vector_store = Chroma(
            client=persistent_client,
            collection_name=self.config.train["model"]["collection_name"],
            embedding_function=self.embeddings,
        )

    def _get_retriever(self, image_class):
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.config.train["output_document_from_vector_store"],
                "filter": {"category": image_class},
            },
        )
        return retriever

    @staticmethod
    def format_docs(docs):
        "\n\n".join(doc.page_content for doc in docs)

    def run(self, query: str, image_class: str):
        logger.debug(f"Image class: {image_class}")
        logger.debug(f"User query: {query}")

        retrived_query = self._get_retriever(image_class=image_class).invoke(query)
        # logger.debug(retrived_query)
        formatted_docs = "\n\n".join(doc.page_content for doc in retrived_query)
        prompt = f"""You are an expert assistant for question answering tasks.
         Use the following context information to answer the question. Only answer once, and always limit your answers to the document content only.
         Never make up answers. If you do not have the answer, state that the data is not contained in your knowledge base and stop your response.
         Question: {query}
         Context:{formatted_docs}"""
        logger.debug(f"Prompt Length: {len(prompt)}")
        # result = self.llm.invoke(prompt)
        final_message = ""
        for chunk in self.llm.stream(prompt):
            final_message += chunk.content
        return final_message


if __name__ == "__main__":
    rp = RagPipeline(rag_config_path=RAG_CONFIG_PATH)
    query = "What is the ram of this model ?"
    image_class = (class_name_mapping_dict["c14"]).strip()
    # while True:
    # result = rp.run(query, image_class)
    # print(result.content)
    for chunk in rp.run(query=query, image_class=image_class):
        print(chunk.content, end="", flush=True)
    # break
    # prompt = "Tell me something about Azure."
    # for chunk in rp.llm.stream(prompt):
    #     print(chunk.content, end="", flush=True)
    # print(rp.llm)
