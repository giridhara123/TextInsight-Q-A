import os
import csv
import warnings
from datetime import datetime
import spacy
from transformers import pipeline
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv

# 🔹 Suppress noisy warnings
from urllib3.exceptions import NotOpenSSLWarning
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint")

# Suppress HuggingFace parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Initialize LLM
try:
    from langchain_openai import OpenAI
    llm = OpenAI(temperature=0.9, max_tokens=500)
except Exception:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(temperature=0.9, model="gpt-4o-mini")

# Initialize spaCy and NLI model
nlp = spacy.load("en_core_web_sm")
nli = pipeline("text-classification", model="roberta-large-mnli", device=-1)  # -1 = CPU

# Configuration
folder_path = "faiss_store_openai_test"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
data_file = "test_data.txt"

def load_test_data(file_path: str) -> tuple[list, list]:
    """Load URLs and queries from an indexed text file."""
    urls = []
    queries = []
    current_section = None
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found. Please create it with [URLs] and [Queries] sections.")
        return [], []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == "[URLs]":
                    current_section = "urls"
                    continue
                if line == "[Queries]":
                    current_section = "queries"
                    continue
                if current_section == "urls" and line[0].isdigit():
                    urls.append(line[line.index(".") + 1:].strip())
                elif current_section == "queries" and line[0].isdigit():
                    queries.append(line[line.index(".") + 1:].strip())
        if not urls or not queries:
            print(f"⚠️ Warning: {file_path} is missing URLs or Queries. Please check formatting.")
        return urls, queries
    except Exception as e:
        print(f"❌ Error reading {file_path}: {str(e)}")
        return [], []

def extract_claims(answer: str, max_claims: int = 6) -> list:
    """Extract factual claims from an answer using spaCy."""
    doc = nlp(answer)
    claims = []
    for sent in doc.sents:
        if "?" in sent.text or any(token.lemma_ in ["think", "believe"] for token in sent):
            continue  # Skip non-factual sentences
        claims.append(sent.text.strip())
        if len(claims) >= max_claims:
            break
    return claims

def detect_hallucinations(claim: str, chunks: list) -> list:
    """Classify claims against chunks using NLI."""
    results = []
    for chunk in chunks:
        output = nli(f"Premise: {chunk.page_content} Hypothesis: {claim}")
        result = output[0]  # pipeline returns a list
        label_map = {"ENTAILMENT": "Supported", "CONTRADICTION": "Contradicted", "NEUTRAL": "NEI"}
        hallucination = result["label"] in ["CONTRADICTION", "NEUTRAL"]
        results.append({
            "claim": claim,
            "label": label_map.get(result["label"], "Unknown"),
            "score": result["score"],
            "chunk": chunk.page_content[:100],  # Short preview for logging
            "hallucination": hallucination
        })
    return results

def main():
    # Load URLs and queries
    test_urls, test_queries = load_test_data(data_file)
    if not test_urls or not test_queries:
        print("❌ No valid test data found. Exiting.")
        return
    print(f"✅ Loaded {len(test_urls)} URLs and {len(test_queries)} queries.")

    # Process URLs
    print("⚙️ Processing URLs...")
    try:
        loader = UnstructuredURLLoader(urls=test_urls)
        data = loader.load()
        if not data:
            print("❌ No content loaded from URLs.")
            return

        splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
        docs = splitter.split_documents(data)
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(folder_path)
    except Exception as e:
        print(f"❌ Error processing URLs: {str(e)}")
        return

    # Initialize RAG chain
    try:
        vectorstore = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    except Exception as e:
        print(f"❌ Error initializing RAG chain: {str(e)}")
        return

    # Process queries and detect hallucinations
    results = []
    for query in test_queries:
        print(f"🔍 Processing query: {query}")
        try:
            result = chain.invoke({"question": query})
            answer = result.get("answer", "")
            chunks = result.get("source_documents", [])
            if not chunks:
                print(f"⚠️ No source documents returned for query: {query}")
                continue

            claims = extract_claims(answer)
            if not claims:
                print(f"⚠️ No claims extracted for query: {query}")
                continue

            for claim in claims:
                nli_results = detect_hallucinations(claim, chunks)
                for nli_result in nli_results:
                    results.append({
                        "query": query,
                        "answer": answer,
                        "claim": nli_result["claim"],
                        "label": nli_result["label"],
                        "score": nli_result["score"],
                        "chunk": nli_result["chunk"],
                        "hallucination": nli_result["hallucination"]
                    })
        except Exception as e:
            print(f"❌ Error processing query '{query}': {str(e)}")

    # Save results to CSV
    output_file = "hallucination_results.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "answer", "claim", "label", "score", "chunk", "hallucination"])
        writer.writeheader()
        writer.writerows(results)

    # Compute and print Hallucination Rate
    total_claims = len(results)
    hallucination_count = sum(1 for r in results if r["hallucination"])
    hallucination_rate = hallucination_count / total_claims if total_claims > 0 else 0
    print(f"\n📊 Hallucination Rate: {hallucination_rate:.2%} ({hallucination_count}/{total_claims} claims)")
    print(f"📁 Results saved to: {output_file}")

if __name__ == "__main__":
    main()