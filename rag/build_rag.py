import json
from langchain_community.document_loaders import PyPDFLoader
from uuid import uuid4
from base import BaseRag

PATH_TO_PDF: str = r"../data/pdfs/c0/all-products_esuprt_desktop_esuprt_alienware_dsk_alienware-alpha_reference guide_en-us.pdf"


class CustomRag(BaseRag):
    def __init__(self, rag_config) -> None:
        super().__init__(rag_config)

    def get_prompt(self) -> str:
        """Take user prompt and sends it to check_cache function."""
        ...

    def check_cache(self, prompt: str):
        """Checks Reddis cache and returns the response. If miss then calls get_response function"""
        ...

    def get_response(self, prompt: str):
        """Use Vector db to fetch the response"""
        ...

    def format_response(self, prompt: str):
        """Format the response with LLM."""
        ...

    def save_cache(self, prompt: str, response: str):
        """Save the prompt and response to the cache."""
        ...


def main(rag_config) -> None:
    loader = PyPDFLoader(PATH_TO_PDF)
    pages = loader.load_and_split()
    cr = CustomRag(rag_config)


if __name__ == "__main__":
    rag_config_json: str = r"../rag_config.json"
    with open(rag_config_json, "r") as f:
        rag_config_dict: dict[str, any] = json.load(f)
    main(rag_config_dict)