import os
from tqdm import tqdm
from pathlib import Path
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

db_name = "vector_db"


def create_chunks(folder_path):
    data_path = Path(folder_path)

    def add_metadata(doc, doc_type):
        doc.metadata["doc_type"] = doc_type
        return doc

    # Get all subdirectories
    folders = [f for f in data_path.iterdir() if f.is_dir()]

    text_loader_kwargs = {"encoding": "utf-8"}
    documents = []

    for folder in folders:
        doc_type = folder.name
        loader = DirectoryLoader(
            str(folder),
            glob="**/*",  # Match all files
            loader_cls=TextLoader,
            loader_kwargs=text_loader_kwargs,
        )
        try:
            folder_docs = loader.load()
            documents.extend(
                [add_metadata(doc, doc_type) for doc in folder_docs]
            )
        except Exception as e:
            print(f"Error loading documents from {folder}: {e}")
            continue

    text_splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Total number of chunks: {len(chunks)}")
    print(
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
    if os.path.exists(db_name):
        print(
            "Vectorstore already exists. Skipping chunk creation and loading existing vectorstore."
        )
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory=db_name, embedding_function=embeddings
        )
        print(
            f"Loaded existing vectorstore with {vectorstore._collection.count()} documents"
        )
    else:
        print("Creating new vectorstore from documents...")
        chunks = create_chunks(folder_path)
        vectorstore = chunk_to_vector(chunks)
    return vectorstore


def chunk_investigator(vectorstore):
    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])[
        "embeddings"
    ][0]
    dimensions = len(sample_embedding)
    print(
        f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store"
    )
    # tsne_visualizer(collection)
