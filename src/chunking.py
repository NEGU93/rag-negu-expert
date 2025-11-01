import os
import hashlib
from tqdm import tqdm
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,
    UnstructuredXMLLoader,
    UnstructuredImageLoader,
    JSONLoader,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.website_scraper import scrape_website, scrape_github
from src.logger_init import logger


db_name = "vector_db"


loaders = {
    ".pdf": PyMuPDFLoader,
    ".xml": UnstructuredXMLLoader,
    ".md": TextLoader,
    ".tex": TextLoader,
    ".txt": TextLoader,
    ".jpg": UnstructuredImageLoader,
    ".json": lambda path: JSONLoader(
        file_path=path, jq_schema=".", text_content=False
    ),
}


def get_data_hash(folder_path="data"):
    hash_md5 = hashlib.md5()
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            path = os.path.join(root, f)
            hash_md5.update(f.encode())
            with open(path, "rb") as file:
                hash_md5.update(file.read())
    return hash_md5.hexdigest()


def load_single_file(file_path, loader_class):
    """Load a single file with proper error handling"""
    try:
        loader = loader_class(str(file_path))
        return loader.load()
    except Exception as e:
        logger.error(f"Failed to load {file_path.name}: {e}")
        return []


def create_chunks(folder_path):
    data_path = Path(folder_path)

    def add_metadata(doc, doc_type):
        doc.metadata["doc_type"] = doc_type
        if "source" in doc.metadata:
            source_path = Path(doc.metadata["source"])
            doc.metadata["filename"] = source_path.name
            doc.metadata["file_path"] = str(source_path)
        return doc

    folders = [f for f in data_path.iterdir() if f.is_dir()]
    documents = []
    file_stats = {"loaded": 0, "skipped": 0, "errors": 0}

    for folder in folders:
        doc_type = folder.name
        logger.info(f"Processing folder: {doc_type}")

        # Find all files in the folder
        files = [f for f in folder.rglob("*") if f.is_file()]

        for file_path in tqdm(
            files, desc=f"Processing {doc_type}", leave=False
        ):
            file_extension = file_path.suffix.lower()

            if file_extension in loaders:
                file_docs = load_single_file(
                    file_path, loaders[file_extension]
                )
                if file_docs:
                    # Add metadata to each document
                    documents.extend(
                        [add_metadata(doc, doc_type) for doc in file_docs]
                    )
                    logger.debug(f"Loaded: {file_path.name}")
                    file_stats["loaded"] += 1
                else:
                    file_stats["errors"] += 1
            else:
                logger.debug(f"Skipped unsupported file: {file_path.name}")
                file_stats["skipped"] += 1

    # Summary
    logger.info("File processing summary:")
    logger.info(f"  - Loaded: {file_stats['loaded']} files")
    logger.info(f"  - Skipped: {file_stats['skipped']} files")
    logger.info(f"  - Errors: {file_stats['errors']} files")

    if not documents:
        logger.warning("No documents found!")
        return []

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    logger.info(f"Total number of documents loaded: {len(documents)}")
    logger.info(f"Total number of chunks: {len(chunks)}")
    logger.info(
        f"Document types found: {set(doc.metadata['doc_type'] for doc in documents)}"
    )

    return chunks


def chunk_to_vector(chunks):
    embeddings = OpenAIEmbeddings()
    if os.path.exists(db_name):
        Chroma(
            persist_directory=db_name, embedding_function=embeddings
        ).delete_collection()

    vectorstore = Chroma(
        persist_directory=db_name, embedding_function=embeddings
    )

    # Process in smaller batches
    batch_size = 50
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i : i + batch_size]
        try:
            vectorstore.add_documents(batch)
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            # Reduce batch size and retry
            for doc in batch:
                try:
                    vectorstore.add_documents([doc])
                except Exception as doc_error:
                    print(f"Skipping problematic document: {doc_error}")
                    continue

    print(
        f"Vectorstore created with {vectorstore._collection.count()} documents"
    )
    return vectorstore


def init_db(folder_path="data", db_name=db_name):
    scrape_website()
    scrape_github()

    hash_file = os.path.join(db_name, ".data_hash")

    current_hash = get_data_hash(folder_path)
    stored_hash = None
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            stored_hash = f.read().strip()
    else:
        logger.info("No existing hash file found.")

    if os.path.exists(db_name) and stored_hash == current_hash:
        logger.info("✅ Vectorstore is up to date. Loading existing one...")
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory=db_name, embedding_function=embeddings
        )
    else:
        logger.info(
            "🛠 Changes detected or DB missing. Rebuilding vectorstore..."
        )
        chunks = create_chunks(folder_path)
        vectorstore = chunk_to_vector(chunks)

        # Save hash for next time
        os.makedirs(db_name, exist_ok=True)
        with open(hash_file, "w") as f:
            f.write(current_hash)

    logger.info(
        f"Vectorstore ready with {vectorstore._collection.count()} documents"
    )
    return vectorstore
