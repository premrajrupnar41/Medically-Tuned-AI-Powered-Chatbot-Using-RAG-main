import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# Load environment variables (HF_TOKEN from .env)
load_dotenv()

# Constants
DB_FAISS_PATH = "DB_FAISS_PATH"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Load LLM from Hugging Face
def load_llm():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        huggingfacehub_api_token=os.environ.get("HF_TOKEN"),
        temperature=0.5,
        max_new_tokens=512
    )

# Set custom prompt template
custom_prompt_template = PromptTemplate(
    template=(
        "You are a professional medical assistant designed to answer only health and medical-related questions.\n"
        "If the user's question is not related to the medical field, kindly respond with:\n"
        "\"I'm sorry, but I can only assist with questions specifically related to medical or health-related topics.\"\n\n"
        "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    ),
    input_variables=["context", "question"]
)

# custom_prompt_template = PromptTemplate(
#     template=(
#         "You are a helpful assistant. Answer the question based on the context provided. "
#         "If you do not know the answer, just say so instead of making something up.\n\n"
#         "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
#     ),
#     input_variables=["context", "question"]
# )

# Load embeddings and FAISS vector store
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 3})

# Run QA
def run_qa():
    retriever = load_vector_store()
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt_template}
    )

    user_query = input("Enter your medical question: ")
    result = qa_chain({"query": user_query})

    print("\n🧠 Answer:", result['result'])
    print("\n📚 Source Documents:")
    for doc in result['source_documents']:
        print("-", doc.metadata.get("source", "Unknown"))
        print(doc.page_content[:300] + "...\n")

if __name__ == "__main__":
    run_qa()

