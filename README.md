This project is a Retrieval-Augmented Generation (RAG) Chatbot developed using Streamlit for the user interface, LangChain for orchestration, HuggingFace Transformers for the language model, and FAISS for vector similarity search.

The chatbot allows users to upload or query information from a custom document and generates responses strictly based on the provided context. It also maintains chat history within the session for more interactive and conversational experiences.
# RAG Chatbot with Streamlit

This project is a Retrieval-Augmented Generation chatbot built using:
- Streamlit
- LangChain
- HuggingFace Transformers
- FAISS

## How to Run

```bash
pip install -r requirements.txt
streamlit run app2.py
