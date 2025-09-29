import os
import streamlit as st
import requests
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

from transformers import pipeline
import spacy

# ---- Setup ----
load_dotenv()
st.set_page_config(page_title="TextInsight Engine", layout="wide")

# Custom styles
st.markdown("""
    <style>
    .big-font { font-size: 24px !important; font-weight: bold; color: #1f77b4; }
    .status-box { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">TextInsight Engine</p>', unsafe_allow_html=True)

# API + FAISS storage
API_URL = "http://localhost:8000"
folder_path = "faiss_store_openai"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# Load LLM
try:
    from langchain_openai import OpenAI
    llm = OpenAI(temperature=0.0, max_tokens=500)  # factual mode
except Exception:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

# NLI + spaCy for hallucination detection
nlp = spacy.load("en_core_web_sm")
nli = pipeline("text-classification", model="roberta-large-mnli", device=-1)


# ---- Utility Functions ----
def extract_claims(answer: str, max_claims: int = 6):
    """Extract factual claims from an answer using spaCy."""
    doc = nlp(answer)
    claims = []
    for sent in doc.sents:
        if "?" in sent.text or any(token.lemma_ in ["think", "believe"] for token in sent):
            continue
        claims.append(sent.text.strip())
        if len(claims) >= max_claims:
            break
    return claims


def detect_hallucination(claim, chunks, threshold=0.65):
    """
    Improved hallucination detection:
    - Aggregate evidence across all chunks.
    - If ANY chunk strongly supports the claim → Supported.
    - Otherwise → Hallucination.
    """
    supported = False
    contradicted = False

    for chunk in chunks:
        output = nli(f"Premise: {chunk.page_content} Hypothesis: {claim}")
        result = output[0]
        label, score = result["label"], result["score"]

        if label == "ENTAILMENT" and score >= threshold:
            supported = True
            break  # no need to check further if we found strong support
        elif label == "CONTRADICTION" and score >= threshold:
            contradicted = True

    if supported:
        return False  # Not a hallucination
    elif contradicted:
        return True   # Direct contradiction
    else:
        return False  # NEUTRAL → treat as "Unverifiable but not hallucination"


# ---- Sidebar: URLs ----
with st.sidebar:
    st.header("Add Article URLs")
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


# ---- Process URLs ----
if process_url_clicked:
    urls = [u.strip() for u in st.session_state.url_list if u.strip()]
    if not urls:
        st.error("Please enter at least one valid URL.")
    else:
        with st.spinner("Processing your URLs..."):
            try:
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()
                if not data:
                    st.warning("No content loaded from the provided URLs.")
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','], chunk_size=1000
                )
                docs = splitter.split_documents(data)

                embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
                vectorstore = FAISS.from_documents(docs, embeddings)
                vectorstore.save_local(folder_path)

                st.success("✅ URLs processed and embeddings saved successfully!")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


# ---- Query ----
st.subheader("Ask a Question")
with st.form(key="query_form", clear_on_submit=True):
    query = st.text_input(
        "Enter your question here:",
        placeholder="e.g., Who founded Zoho Corporation?",
        key="query_input"
    )
    submit_button = st.form_submit_button(label="Get Answer")

if submit_button and query:
    if not os.path.exists(folder_path):
        st.error("No embeddings found. Please process URLs first.")
    else:
        try:
            with st.spinner("Generating answer..."):
                embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
                vectorstore = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)

                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=llm, chain_type="map_reduce", retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
                    return_source_documents=True
                )
                result = chain.invoke({"question": query})

                st.markdown("### Answer")
                st.write(result["answer"])

                hallucinations = []
                claims = extract_claims(result["answer"])
                for claim in claims:
                    if detect_hallucination(claim, result.get("source_documents", [])):
                        hallucinations.append(claim)

                if hallucinations:
                    st.warning("⚠️ Possible Hallucinations Detected:")
                    for h in hallucinations:
                        st.markdown(f"- {h}")

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


# ---- History ----
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