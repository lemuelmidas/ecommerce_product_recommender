import streamlit as st
import polars as pl
import re
import unicodedata
from dateutil import parser as du_parser
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import time

# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(page_title="E-commerce Vector Search", layout="wide")

# ------------------------------
# Load Embedding Model (Local)
# ------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()  # Embedding dimension = 384

# ------------------------------
# UTILS
# ------------------------------
def sanitize_text(s: str, max_len=None):
    if s is None:
        return None
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"[\x00-\x1F\x7F]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if max_len and len(s) > max_len:
        return s[:max_len]
    return s

def safe_float(x):
    try:
        x = re.sub(r"[^\d\.\-]", "", str(x))
        return float(x) if x else None
    except:
        return None

def chunk_text(text: str, chars_per_chunk=2000, overlap=200):
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chars_per_chunk)
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def embed_texts(texts):
    return model.encode(texts).tolist()

# Simple PII detection
PII_PATTERNS = [
    re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"),  
    re.compile(r"\b4[0-9]{12}(?:[0-9]{3})?\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
]

def contains_pii(s):
    return any(p.search(s) for p in PII_PATTERNS)

# ------------------------------
# Pinecone Helper
# ------------------------------
def get_or_create_index(pc: Pinecone, index_name: str, dim: int = 384, metric: str = "cosine", region: str = "us-east-1"):
    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region=region)
        )
        st.info(f"Creating index '{index_name}', please wait...")

    # Poll until index is ready
    while index_name not in [i.name for i in pc.list_indexes()]:
        st.info("Waiting for Pinecone index to be ready...")
        time.sleep(2)

    return pc.Index(index_name)

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("🛒 E-commerce Product Vector Search & Recommendation")

st.markdown("""
Upload your CSV → Clean it → Build vectors → Send to Pinecone → Query recommendations.
""")

# ---------------------------------
# 1. CSV UPLOAD
# ---------------------------------
uploaded = st.file_uploader("Upload your E-commerce CSV", type=["csv"])

if uploaded:
    st.success("CSV uploaded!")

    st.subheader("🔧 Data Cleaning")

    df = pl.read_csv(uploaded, ignore_errors=True)

    st.write("Raw Data Sample:")
    st.dataframe(df.head())

    df = (
        df.with_columns([
            pl.col("Description").apply(sanitize_text, return_dtype=pl.Utf8).alias("Description"),
            pl.col("InvoiceDate").apply(lambda x: du_parser.parse(x) if x else None, return_dtype=pl.Datetime).alias("InvoiceDate"),
            pl.col("UnitPrice").apply(safe_float, return_dtype=pl.Float64).alias("UnitPrice"),
            pl.col("Quantity").cast(pl.Int64).alias("Quantity"),
            pl.col("CustomerID").apply(safe_float, return_dtype=pl.Float64).alias("CustomerID"),
            pl.col("Country").apply(sanitize_text, return_dtype=pl.Utf8).alias("Country")
        ])
        .unique()
    )

    st.write("Cleaned Data Sample:")
    st.dataframe(df.head())

    st.session_state["clean_df"] = df

# ---------------------------------
# 2. Build Vector DB
# ---------------------------------
st.subheader("📦 Vector Database Builder")

pinecone_key = st.text_input("Pinecone API Key", type="password")
pinecone_index = st.text_input("Pinecone Index Name", "products-vec-db")

if st.button("Build Vector Database"):
    if "clean_df" not in st.session_state:
        st.error("Upload and clean a CSV first.")
    elif not pinecone_key:
        st.error("Enter Pinecone API key.")
    else:
        df = st.session_state["clean_df"]
        pc = Pinecone(api_key=pinecone_key)
        index = get_or_create_index(pc, pinecone_index)

        st.info("Chunking & embedding... (may take a while)")

        all_records = []
        for row in df.to_dicts():
            product_id = f"{row.get('InvoiceNo', '')}_{row.get('StockCode', '')}"
            text = row.get("Description") or ""
            chunks = chunk_text(text)

            for i, ch in enumerate(chunks):
                all_records.append({
                    "id": f"{product_id}__chunk{i}",
                    "text": ch,
                    "metadata": {
                        "product_id": product_id,
                        "snippet": ch[:150],
                        "unit_price": row.get("UnitPrice"),
                        "country": row.get("Country")
                    }
                })

        BATCH = 128
        for i in range(0, len(all_records), BATCH):
            batch = all_records[i:i + BATCH]
            texts = [r["text"] for r in batch]
            vectors = embed_texts(texts)

            pinecone_items = [
                (batch[j]["id"], vectors[j], batch[j]["metadata"])
                for j in range(len(batch))
            ]
            index.upsert(pinecone_items)

            st.write(f"Inserted {i + len(batch)} / {len(all_records)} vectors...")

        st.success("Vector DB build complete!")

# ---------------------------------
# 3. Search / Recommendation
# ---------------------------------
st.subheader("🔍 Product Recommendation")

query = st.text_area("Enter a natural language query", placeholder="Example: 'I want white lantern home decor'")
top_k = st.slider("Top-K Results", 1, 10, 5)

if st.button("Search"):
    if not pinecone_key:
        st.error("Enter Pinecone API key!")
    elif contains_pii(query):
        st.error("Query contains PII. Blocked.")
    elif len(query.strip()) == 0:
        st.error("Enter a valid query.")
    else:
        pc = Pinecone(api_key=pinecone_key)
        index = get_or_create_index(pc, pinecone_index)

        q_vec = embed_texts([query])[0]

        res = index.query(vector=q_vec, top_k=top_k, include_metadata=True)

        st.write("### Results")

        products = {}
        for m in res["matches"]:
            pid = m["metadata"]["product_id"]
            if pid not in products:
                products[pid] = {
                    "score": 0,
                    "snippet": m["metadata"]["snippet"]
                }
            products[pid]["score"] += m["score"]

        ranked = sorted(products.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]

        for pid, info in ranked:
            snippet = info['snippet'][:200]
            st.markdown(
                f"**Product ID:** {pid}\n\n"
                f"**Score:** {info['score']:.4f}\n\n"
                f"**Snippet:** {snippet}\n\n"
                "---"
            )
