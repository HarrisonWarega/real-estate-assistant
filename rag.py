# ============================
# CORE IMPORTS
# ============================
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from uuid import uuid4  # unique IDs for vector DB chunks
from dotenv import load_dotenv  # environment variables
from pathlib import Path

# document loading
from langchain_community.document_loaders import UnstructuredURLLoader
# text chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
# vector database
from langchain_chroma import Chroma
# embeddings model
from langchain_huggingface import HuggingFaceEmbeddings
# Groq LLM
from langchain_groq import ChatGroq
# modern retrieval chain
from langchain.chains import RetrievalQA

# ============================
# LOAD ENV VARIABLES
# ============================
load_dotenv()

# ============================
# CONSTANTS (configuration)
# ============================
CHUNK_SIZE = 1000  # size of text blocks for embedding
CHUNK_OVERLAP = 200  # prevents context fragmentation

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

# ============================
# GLOBAL STATE (keeps objects alive across Streamlit reruns)
# ============================
llm = None
vector_store = None
qa_chain = None

# ============================
# INITIALIZATION FUNCTION
# ============================

def initialize_components():
    global llm, vector_store

    # STEP 1: initialize LLM only once
    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",  # stable production model
            temperature=0.2
        )

    # STEP 2: initialize embeddings + vector DB only once
    if vector_store is None:

        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        # embedding model converts text → vectors
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        # Chroma vector DB (persistent storage)
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTORSTORE_DIR)
        )

# ============================
# PROCESS URLs → BUILD KNOWLEDGE BASE
# ============================
def process_urls(urls):

    global qa_chain

    # STEP 1: initialize LLM + vector DB
    yield "Initializing components..."
    initialize_components()

    # STEP 2: reset vector DB (fresh ingestion each run)
    yield "Resetting vector database..."
    vector_store.reset_collection()

    # STEP 3: load web pages
    yield "Loading URLs..."
    loader = UnstructuredURLLoader(
        urls=urls,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }      
    )
    docs = loader.load()

    # STEP 4: split documents into chunks
    yield "Splitting documents..."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    # STEP 5: convert chunks → embeddings → store in DB
    yield "Storing embeddings..."
    vector_store.add_documents(
        documents=chunks,
        ids=[str(uuid4()) for _ in chunks]
    )
    yield "Ready!"

# ================================================
# QUERY PIPELINE (RETRIEVAL + RERANK + GROUNDED QA)
# ================================================

def generate_answer(query):

    global qa_chain

    # safety check
    if vector_store is None:
        raise RuntimeError("Run process_urls first")

    # create retriever from vector DB (MMR reduces redundancy)
    retriever = vector_store.as_retriever(
        search_type="mmr",  # diversity-aware retrieval
        search_kwargs={"k": 5}
    )

    # build chain ONLY ONCE
    if qa_chain is None:

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": None  # Rely on default but can extend later
            }
        )

    # run retrieval + generation
    result = qa_chain.invoke({"query": query})

    # extract answer
    answer = result["result"]

    # extract sources
    sources = result.get("source_documents", [])
    
# =============================
#  CLEAN + DEDUPLICATE SOURCES
# =============================

    seen = set()
    unique_sources = []
    
    for doc in sources:
        src = doc.metadata.get("source", "unknown")
    if src not in seen:
        seen.add(src)
        unique_sources.append(src)
    
    formatted_sources = "\n".join(unique_sources)

# ======================
# DEBUG TRACE (optional)
# ======================
    
    for i, doc in enumerate(sources):
        print(f"\n--- CHUNK {i} ---")
        print(doc.page_content[:300])
    
    return answer, formatted_sources
    
if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    for status in process_urls(urls):
        print(status)

    answer, sources = generate_answer(
        "Tell me what was the 30 year fixed mortgage rate along with the date?"
    )

    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
