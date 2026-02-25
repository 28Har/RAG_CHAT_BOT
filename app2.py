import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline

# Streamlit UI

st.set_page_config(page_title="WELCOME TO ECI CHAT", layout="centered")
st.title("Welcome to ECI")

# Initialize Chat History 

if "history" not in st.session_state:
    st.session_state["history"] = []


# Load document

loader = TextLoader("data/ECI.txt", encoding="utf-8")
documents = loader.load()


# Split text

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)


# Embeddings & Vector Store

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()


# HuggingFace Model

hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=256,
    temperature=0.2
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Prompt Template

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer ONLY from the given context.
If the answer is not in the context, say "I don't know from the document."

Context:
{context}

Chat History:
{history}

Question:
{question}
""")


# Helper functions

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history():
    history = st.session_state.get("history", [])  # SAFE access
    text = ""
    for h in history:
        text += f"User: {h['user']}\nAI: {h['ai']}\n"
    return text


# Chain

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        "history": lambda x: format_history()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# UI Chat Input

user_input = st.chat_input("Ask from your document...")

# Display previous chat history
for h in st.session_state.history:
    st.chat_message("user").write(h["user"])
    st.chat_message("assistant").write(h["ai"])


# When user asks question

if user_input:
    st.chat_message("user").write(user_input)

    response = chain.invoke(user_input)

    st.chat_message("assistant").write(response)

    st.session_state.history.append(
        {"user": user_input, "ai": response}
    )