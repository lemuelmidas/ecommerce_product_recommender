import streamlit as st
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pickle

############################################
# ---------- DATA CLEANING UTILS ----------
############################################

def clean_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        # Clean Quantity
        pl.col("Quantity")
            .cast(pl.Utf8, strict=False)
            .str.replace_all(r"[^0-9]", "")
            .cast(pl.Int64, strict=False)
            .fill_null(0)
            .alias("Quantity"),

        # Clean UnitPrice or Price
        pl.col("UnitPrice")
            .cast(pl.Utf8, strict=False)
            .str.replace_all(r"[^0-9.]", "")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias("UnitPrice"),

        # Clean Description (remove weird characters)
        pl.col("Description")
            .cast(pl.Utf8, strict=False)
            .str.replace_all(r"[^A-Za-z0-9 ,.-]", " ")
            .str.strip_chars()
            .alias("Description"),

        # Clean InvoiceNo
        pl.col("InvoiceNo")
            .cast(pl.Utf8, strict=False)
            .str.replace_all(r"[^0-9]", "")
            .alias("InvoiceNo"),

        # Clean CustomerID
        pl.col("CustomerID")
            .cast(pl.Utf8, strict=False)
            .str.replace_all(r"[^0-9]", "")
            .cast(pl.Int64, strict=False)
            .alias("CustomerID")
    ])

############################################
# ---------- CNN MODEL DEFINITION ----------
############################################

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 32 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        emb = x.clone()     # Embedding for similarity search
        out = self.classifier(x)
        return out, emb

############################################
# ---------- MODEL & VECTOR UTILS ----------
############################################

def load_model(model_path, num_classes):
    model = SimpleCNN(num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def preprocess_image(image):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

def load_vectors(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def find_similar(embedding, vector_db, k=5):
    vectors = np.array([item["embedding"] for item in vector_db])
    sims = vectors @ embedding / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(embedding)
    )
    top_idx = np.argsort(sims)[-k:][::-1]
    return [vector_db[i] for i in top_idx]

############################################
# --------------- STREAMLIT UI ------------
############################################

st.title("🛒 Product Image Detection + Recommender (CNN + Vector DB)")

uploaded_file = st.file_uploader("Upload a product image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=350)

    # Load model
    num_classes = 10  # change based on your dataset
    model = load_model("models/cnn_product_model.pth", num_classes)

    # Preprocess
    img_tensor = preprocess_image(image)

    # Predict
    with torch.no_grad():
        outputs, embeddings = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        class_id = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0][class_id].item())

    st.subheader("🔍 CNN Prediction")
    st.write(f"**Predicted Class:** `{class_id}`")
    st.write(f"**Confidence:** `{confidence:.2f}`")

    # Similar products
    vector_db = load_vectors("vector_db/product_vectors.pkl")
    similar = find_similar(embeddings[0].numpy(), vector_db)

    st.subheader("🛍 Related Products (Vector Similarity)")
    for item in similar:
        st.write(f"- **{item['name']}** — {item['description']}")

############################################
# --------------- LOAD CSV ----------------
############################################

st.subheader("📦 Load and Clean Product Dataset")

file = st.file_uploader("Upload CSV for Cleaning", type=["csv"])

if file:
    raw_df = pl.read_csv(file)
    st.write("### Raw Data")
    st.dataframe(raw_df.head())

    cleaned_df = clean_dataframe(raw_df)
    st.write("### Cleaned Data")
    st.dataframe(cleaned_df.head())
