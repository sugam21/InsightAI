import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from uuid import uuid4
from langchain_ollama import OllamaEmbeddings



PATH_TO_PDF: str = r"../data/pdfs/c0/all-products_esuprt_desktop_esuprt_alienware_dsk_alienware-alpha_reference guide_en-us.pdf"

embeddings = OllamaEmbeddings(
    model="all-minilm",
)
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
)


def main(rag_config) -> None:
    loader = PyPDFLoader(PATH_TO_PDF)
    pages = loader.load_and_split()
    uids = [str(uuid4()) for _ in pages]
    vector_store.add_documents(documents=pages, uids=uids)


if __name__ == "__main__":
    rag_config_json: str = r"../rag_config.json"
    with open(rag_config_json, "r") as f:
        rag_config_dict: dict[str, any] = json.load(f)
    main(rag_config_dict)
    results = vector_store.similarity_search(
        "What is the input voltage frequency?",
        k=1,
    )
    for res in results:
        print(f"* {res.page_content}")