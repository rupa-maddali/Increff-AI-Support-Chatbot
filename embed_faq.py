"""
Run once after editing data/faq.json:
    python embed_faq.py
"""

import json
import pickle
import os
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
EMBED_MODEL = "text-embedding-3-small"
DATA_PATH = "data/faq.json"
INDEX_PATH = "data/faq_faiss.index"
META_PATH = "data/faq_meta.pkl"


os.environ["USER_AGENT"] = "MyApp/1.0 (+https://example.com)"
os.environ[
    "OPENAI_API_KEY"] = "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"  # Replace with your actual key



def embed(texts: list[str], client: OpenAI) -> np.ndarray:
    """Embed texts using OpenAI's API with proper error handling and batching."""
    vectors = []
    for i in range(0, len(texts), 100):  # Batch size of 100
        try:
            chunk = texts[i:i + 100]
            res = client.embeddings.create(model=EMBED_MODEL, input=chunk)
            vectors.extend([d.embedding for d in res.data])
        except Exception as e:
            print(f"Error embedding batch {i // 100}: {str(e)}")
            raise

    if not vectors:
        raise ValueError("No vectors were generated")

    vecs = np.array(vectors, dtype="float32")
    # Normalize for cosine similarity (≈ inner product)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.clip(norms, 1e-8, None)  # Avoid division by zero
    return vecs


def main():
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load and validate FAQ data
    try:
        with open(DATA_PATH) as f:
            faq = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading FAQ data: {str(e)}")
        return

    if not faq:
        print("Error: FAQ data is empty")
        return

    questions, answers = zip(*faq.items())

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(INDEX_PATH) or os.path.dirname(INDEX_PATH), exist_ok=True)

    # Generate embeddings and build index
    try:
        vecs = embed(list(questions), client)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)

        # Save the index and metadata
        faiss.write_index(index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump({"questions": questions, "answers": answers}, f)

        print(f"✅ Successfully indexed {len(questions)} FAQ rows")
        print(f"   - Index saved to {INDEX_PATH}")
        print(f"   - Metadata saved to {META_PATH}")

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        # Clean up partial files if they exist
        for path in [INDEX_PATH, META_PATH]:
            if os.path.exists(path):
                os.remove(path)
        return


if __name__ == "__main__":
    main()