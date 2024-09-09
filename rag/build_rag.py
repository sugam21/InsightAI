import json
from langchain_community.document_loaders import PyPDFLoader
from uuid import uuid4
from base import BaseRag
from logger import get_logger

PATH_TO_PDF: str = r"../data/pdfs/c0/all-products_esuprt_desktop_esuprt_alienware_dsk_alienware-alpha_reference guide_en-us.pdf"
LOG = get_logger("build_rag")

class CustomRag(BaseRag):
    def __init__(self, rag_config) -> None:
        super().__init__(rag_config)

    def ask_question(self) -> str:
        """Take user prompt and sends it to check_cache function."""
        while True:
            user_question: str = input("Ask me anything about the pdf >>")

            if user_question.lower in ['exit', 'quit']:
                break
            response: str | None = self.check_cache(user_question)

    def check_cache(self, user_prompt: str) -> str | None:
        """Checks Reddis cache and returns the response. If miss then calls get_response function"""
        # Check the cache for a response
        if response := self.cache.check(user_prompt):
            LOG.info(f"Cache hit for query: '{user_prompt}'")
            return response[0]["response"]
        else:
            self.get_response(user_prompt)

    def get_response(self, user_prompt: str):
        """Use Vector db to fetch the response"""
        LOG.info(f"Cache miss for query {user_prompt}")
        ...

    def format_response(self, prompt: str):
        """Format the response with LLM."""
        ...

    def save_cache(self, prompt: str, response: str):
        """Save the prompt and response to the cache."""
        self.cache.store(prompt=prompt, response=response)


def main(rag_config) -> None:
    loader = PyPDFLoader(PATH_TO_PDF)
    pages = loader.load_and_split()
    cr = CustomRag(rag_config)


if __name__ == "__main__":
    rag_config_json: str = r"../rag_config.json"
    with open(rag_config_json, "r") as f:
        rag_config_dict: dict[str, any] = json.load(f)
    main(rag_config_dict)