from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load all PDFs from a directory
pdf_dir = "D:\Projects\Medically Tuned AI Powered Chatbot using RAG\Data"  # Update path as needed

loader = DirectoryLoader(
    pdf_dir,
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()

# 2. Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split the documents into chunks
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings for the chunks
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store embeddings in FAISS
DB_FAISS_PATH= "DB_FAISS_PATH"
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. (Optional) Save the FAISS index to disk

vectorstore.save_local("DB_FAISS_PATH")
