import streamlit as st
import os
from dotenv import load_dotenv
import re

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "DB_FAISS_PATH"

# Function to check if the question is medical-related
def is_medical_question(question):
    medical_keywords = [
        "symptom", "treatment", "medicine", "disease", "health", "illness",
        "diabetes", "cancer", "infection", "blood", "pain", "insulin", "injury",
        "dose", "side effect", "fever", "headache", "vomiting", "therapy", "cure",
        "fracture", "prescription", "pharmacy", "diagnosis", "surgery", "doctor",
        "nurse", "hospital", "pressure", "tablet", "mental", "anxiety", "depression"
    ]
    return any(re.search(rf"\b{kw}\b", question.lower()) for kw in medical_keywords)

# Cache vector store
@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# Load LLM
def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=512
    )

# Main app
def main():
    st.set_page_config(page_title="CuraMind", layout="centered")
    st.title("🩺 CuraMind: Medically Tuned AI Powered Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    prompt = st.chat_input("Ask a question about medical topics:")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not is_medical_question(prompt):
            response = "I'm sorry, but I can only assist with questions specifically related to medical or health-related topics."
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            try:
                retriever = get_vectorstore().as_retriever(search_kwargs={"k": 3})
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(),
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={
                        "prompt": PromptTemplate(
                            template=(
                                "You are a helpful assistant. Answer the question based on the context provided. "
                                "If you do not know the answer, say you don't know.\n\n"
                                "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                            ),
                            input_variables=["context", "question"]
                        )
                    }
                )

                result = qa_chain.invoke({"query": prompt})
                answer = result['result']
                st.session_state.messages.append({"role": "assistant", "content": answer})

                with st.expander("📚 Source Documents"):
                    for doc in result["source_documents"]:
                        st.markdown(f"- **{doc.metadata.get('source', 'Unknown')}**")
                        st.markdown(doc.page_content[:300] + "...")

            except Exception as e:
                error_msg = f"⚠️ An error occurred: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Display chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

if __name__ == "__main__":
    main()