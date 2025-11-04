# =============================================
# RAG HYBRID RETRIEVAL (Fine-Tuned + OpenAI)
# =============================================

import numpy as np
from openai import OpenAI

client = OpenAI()

# Get embedding from your fine-tuned (SentenceTransformer) model
def get_custom_embedding(text, model):
    return model.encode(text, normalize_embeddings=True)  # shape (384,)

# Get embedding from OpenAI text-embedding-3-large
def get_openai_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(response.data[0].embedding)  # shape (1536,)

# Combine embeddings via concatenation + normalization
def get_hybrid_embedding(text, model, alpha=0.5):
    emb_custom = get_custom_embedding(text, model)
    emb_openai = get_openai_embedding(text)

    # Weighted concatenation
    emb_custom_scaled = alpha * emb_custom
    emb_openai_scaled = (1 - alpha) * emb_openai

    hybrid = np.concatenate([emb_custom_scaled, emb_openai_scaled])
    hybrid = hybrid / np.linalg.norm(hybrid)
    return hybrid

# Search through your local syllabus documents
def search_documents(query, documents, model, top_k=5):
    query_embedding = get_hybrid_embedding(query, model)

    # Compute cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    scores = []
    for doc in documents:
        doc_embedding = doc["embedding"]  # must also be hybrid embeddings
        score = cosine_similarity(query_embedding, doc_embedding)
        scores.append((score, doc))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scores[:top_k]]

# Answer construction logic
def answer_question(query, documents, model):
    relevant_docs = search_documents(query, documents, model)
    answer_text = "\n\n".join([doc["text"] for doc in relevant_docs])
    sources = [doc["source"] for doc in relevant_docs]
    return answer_text, sources

# Main loop
def main():
    # Load pre-embedded documents (each must store 'embedding' as hybrid)
    documents = load_cached_documents("cache/")  # your helper function
    model = load_fine_tuned_model("models/fine_tuned_miniLM/")

    print("============================================================")
    print("🎓 SYLLABUS RAG (Hybrid + Fine-Tuned Embeddings)")
    print("============================================================")
    print("📊 Retrieval Mode: HYBRID (Concatenation)")
    print(f"✅ Loaded {len(documents)} documents from cache\n")

    while True:
        q = input("❓ Question (or 'quit'): ").strip()
        if q.lower() in {"quit", "exit"}:
            print("👋 Goodbye!")
            break

        print("\n🔍 Searching...")
        answer, sources = answer_question(q, documents, model)

        print(f"\n💡 Answer:\n{answer}")
        print(f"\n📚 Sources: {', '.join(sources)}")
        print("-" * 60)


if __name__ == "__main__":
    main()
