import os
import streamlit as st
import requests
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

# If your langchain-openai install uses OpenAI() it’s fine; if not, use ChatOpenAI()
try:
    from langchain_openai import OpenAI  # older/simple LLM interface
    LLM_CLS = OpenAI
    llm_kwargs = dict(temperature=0.9, max_tokens=500)
except Exception:
    from langchain_openai import ChatOpenAI  # chat interface
    LLM_CLS = ChatOpenAI
    llm_kwargs = dict(temperature=0.9, model="gpt-4o-mini")

# ----- Config -----
load_dotenv()
st.set_page_config(page_title="TextInsight Engine", layout="wide")

st.markdown("""
    <style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
        color: #1f77b4;
    }
    .status-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">TextInsight Engine</p>', unsafe_allow_html=True)

API_URL = "http://localhost:8000"
folder_path = "faiss_store_openai"
llm = LLM_CLS(**llm_kwargs)
status_placeholder = st.empty()

# ----- Sidebar: URLs -----
with st.sidebar:
    st.header("Add  Article URLs")
    if "url_list" not in st.session_state:
        st.session_state.url_list = [""] * 3

    for i in range(len(st.session_state.url_list)):
        st.session_state.url_list[i] = st.text_input(
            f"URL {i+1}", value=st.session_state.url_list[i], key=f"url_{i}"
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add URL Field", key="add_url"):
            st.session_state.url_list.append("")
            st.rerun()
    with col2:
        if st.button("Clear URLs", key="clear_urls") and len(st.session_state.url_list) > 1:
            st.session_state.url_list = [""]
            st.rerun()

    process_url_clicked = st.button("Process URLs", type="primary", key="process_urls")

# ----- Process URLs -> Build FAISS -----
if process_url_clicked:
    urls = [u.strip() for u in st.session_state.url_list if u.strip()]
    if not urls:
        status_placeholder.error("Please enter at least one valid URL.")
    else:
        with st.spinner("Processing your URLs..."):
            try:
                status_placeholder.text("Data Loading...Started...✅✅✅")
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()
                if not data:
                    status_placeholder.warning("No content loaded from the provided URLs.")
                    st.stop()

                status_placeholder.text("Text Splitter...Started...✅✅✅")
                splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','], chunk_size=1000
                )
                docs = splitter.split_documents(data)

                status_placeholder.text("Embedding Vector Started Building...✅✅✅")
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(docs, embeddings)
                vectorstore.save_local(folder_path)

                status_placeholder.success("URLs processed and embeddings saved successfully! ✅")
            except Exception as e:
                status_placeholder.error(f"An error occurred: {str(e)}")

# ----- Query -----
st.subheader("Ask a Question")
with st.form(key="query_form", clear_on_submit=True):
    query = st.text_input(
        "Enter your question here:",
        placeholder="e.g., ?",
        key="query_input"
    )
    submit_button = st.form_submit_button(label="Get Answer")

if submit_button and query:
    if not os.path.exists(folder_path):
        st.error("No embeddings found. Please process URLs first.")
    else:
        try:
            with st.spinner("Generating answer..."):
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)

                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
                )
                result = chain.invoke({"question": query})

                st.markdown("### Answer")
                st.write(result["answer"])
                if result.get("sources"):
                    with st.expander("Sources", expanded=False):
                        for source in result["sources"].split("\n"):
                            st.markdown(f"- {source}")

                # Save to history via FastAPI
                search_data = {
                    "question": query,
                    "answer": result["answer"],
                    "sources": result.get("sources", ""),
                }
                try:
                    requests.post(f"{API_URL}/searches/", json=search_data, timeout=5)
                except requests.RequestException as e:
                    st.warning(f"Failed to save search history: {str(e)}")
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

# ----- History -----
st.subheader("Previous Searches")
try:
    resp = requests.get(f"{API_URL}/searches/", timeout=5)
    if resp.status_code == 200:
        searches = resp.json()
        for s in searches:
            with st.expander(f"Question: {s['question']} ({s['timestamp']})"):
                st.markdown(f"**Answer**: {s['answer']}")
                if s.get("sources"):
                    st.markdown("**Sources**:")
                    for src in s["sources"].split("\n"):
                        st.markdown(f"- {src}")
    else:
        st.warning("Could not fetch search history.")
except requests.RequestException:
    st.warning("Backend server not running. Start the FastAPI server to view history.")