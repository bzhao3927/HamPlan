#!/usr/bin/env python3
"""
RAG System with Optional Fine-Tuned Embeddings
Can use either base SentenceTransformer or your fine-tuned version
"""

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
import os
import pickle
from pathlib import Path
import PyPDF2
import time
import argparse

# Load environment variables
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# Configuration
USE_HYBRID = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================
# EMBEDDING FUNCTIONS
# =============================================
def get_custom_embedding(text, model):
    return model.encode(text, normalize_embeddings=True)

def get_openai_embedding(text, retry_count=3):
    """Get OpenAI embedding with retry logic."""
    for attempt in range(retry_count):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=text[:8000]
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            if attempt < retry_count - 1:
                print(f"\n    ⚠️  OpenAI API error (attempt {attempt+1}/{retry_count}): {e}")
                time.sleep(2 ** attempt)
            else:
                print(f"\n    ❌ OpenAI API failed after {retry_count} attempts: {e}")
                raise

def get_hybrid_embedding(text, model, alpha=0.5):
    emb_custom = get_custom_embedding(text, model)
    
    if USE_HYBRID:
        try:
            emb_openai = get_openai_embedding(text)
            emb_custom_scaled = alpha * emb_custom
            emb_openai_scaled = (1 - alpha) * emb_openai
            hybrid = np.concatenate([emb_custom_scaled, emb_openai_scaled])
        except Exception as e:
            print(f"\n    ⚠️  Falling back to local embedding only")
            hybrid = emb_custom
    else:
        hybrid = emb_custom
    
    hybrid = hybrid / np.linalg.norm(hybrid)
    return hybrid

# =============================================
# DOCUMENT PROCESSING
# =============================================
def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"⚠️  Error reading {pdf_path}: {e}")
    return text

def find_all_pdfs(syllabi_folder):
    """Find all PDF files in all subfolders."""
    pdf_files = []
    syllabi_path = Path(syllabi_folder)
    
    if not syllabi_path.exists():
        print(f"❌ Folder not found: {syllabi_folder}")
        return []
    
    for subject_folder in syllabi_path.iterdir():
        if subject_folder.is_dir():
            for pdf_file in subject_folder.glob("*.pdf"):
                pdf_files.append(pdf_file)
    
    return pdf_files

def create_embeddings_from_folder(syllabi_folder, model):
    """Extract text from all PDFs and create embeddings."""
    pdf_files = find_all_pdfs(syllabi_folder)
    
    if not pdf_files:
        print(f"❌ No PDF files found in {syllabi_folder}")
        return []
    
    print(f"📚 Found {len(pdf_files)} PDF files")
    print(f"🔧 Mode: {'HYBRID (Local + OpenAI)' if USE_HYBRID else 'LOCAL ONLY'}\n")
    
    all_documents = []
    
    for pdf_idx, pdf_path in enumerate(pdf_files, 1):
        subject = pdf_path.parent.name
        pdf_name = pdf_path.name
        
        print(f"📄 [{pdf_idx}/{len(pdf_files)}] Processing: {subject}/{pdf_name}")
        
        text = extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            print(f"  ⚠️  No text extracted from {pdf_name}")
            continue
        
        chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
        valid_chunks = [c for c in chunks if len(c) >= 50]
        print(f"  📊 {len(valid_chunks)} valid chunks")
        
        print(f"  🔄 Creating embeddings...", end='')
        for i, chunk in enumerate(valid_chunks):
            try:
                embedding = get_hybrid_embedding(chunk, model)
                
                all_documents.append({
                    "text": chunk,
                    "embedding": embedding,
                    "source": f"{subject}/{pdf_name} (chunk {i+1})",
                    "subject": subject,
                    "filename": pdf_name
                })
                
                if (i + 1) % 10 == 0:
                    print(f" {i+1}/{len(valid_chunks)}...", end='')
                    
            except Exception as e:
                print(f"\n    ❌ Error on chunk {i+1}: {e}")
                continue
        
        doc_count = len([d for d in all_documents if d['filename'] == pdf_name])
        print(f" ✅ {doc_count} embeddings\n")
    
    print(f"🎉 Total documents created: {len(all_documents)}")
    return all_documents

def save_documents(documents, cache_path):
    """Save documents to cache."""
    cache_dir = Path(cache_path)
    cache_dir.mkdir(exist_ok=True)
    
    cache_file = cache_dir / "documents.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(documents, f)
    
    print(f"💾 Saved embeddings to {cache_file}")

def load_cached_documents(cache_path):
    """Load pre-embedded documents from cache directory."""
    cache_file = Path(cache_path) / "documents.pkl"
    
    if not cache_file.exists():
        return None
    
    with open(cache_file, "rb") as f:
        documents = pickle.load(f)
    
    return documents

# =============================================
# SEARCH FUNCTIONS
# =============================================
def search_documents(query, documents, model, top_k=5):
    query_embedding = get_hybrid_embedding(query, model)
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    scores = []
    for doc in documents:
        doc_embedding = doc["embedding"]
        score = cosine_similarity(query_embedding, doc_embedding)
        scores.append((score, doc))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scores[:top_k]]

def answer_question(query, documents, model):
    """Search for relevant docs and use GPT to generate an answer."""
    relevant_docs = search_documents(query, documents, model)
    
    context = "\n\n---\n\n".join([
        f"From {doc['source']}:\n{doc['text']}" 
        for doc in relevant_docs
    ])
    
    prompt = f"""Based on the following syllabus information, answer the question concisely and accurately.

Question: {query}

Relevant syllabus excerpts:
{context}

Please provide a clear, direct answer to the question based only on the information provided above."""

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about college syllabi. Be concise and accurate."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        sources = [doc["source"] for doc in relevant_docs]
        
        return answer, sources
        
    except Exception as e:
        print(f"Error getting GPT response: {e}")
        answer_text = "\n\n".join([doc["text"] for doc in relevant_docs])
        sources = [doc["source"] for doc in relevant_docs]
        return answer_text, sources

# =============================================
# MAIN
# =============================================
def main():
    parser = argparse.ArgumentParser(description="RAG System for Syllabus Q&A")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Path to SentenceTransformer model (base or fine-tuned)"
    )
    parser.add_argument(
        "--syllabi-folder",
        type=str,
        default="/Users/CS/Documents/Deep Learning/Final Project/syllabi",
        help="Path to syllabi folder"
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="cache/",
        help="Path to cache directory"
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"📥 Loading model: {args.model}")
    if Path(args.model).exists():
        print("   (Using fine-tuned model)")
        model_type = "FINE-TUNED"
    else:
        print("   (Using base model)")
        model_type = "BASE"
    
    model = SentenceTransformer(args.model)
    
    # Check for cached documents
    documents = load_cached_documents(args.cache_path)
    
    # If no cache, create embeddings
    if documents is None:
        print("\n⚠️  No cached embeddings found. Creating new embeddings...")
        
        if not Path(args.syllabi_folder).exists():
            print(f"❌ Syllabi folder not found: {args.syllabi_folder}")
            return
        
        documents = create_embeddings_from_folder(args.syllabi_folder, model)
        
        if not documents:
            print("❌ No documents were created. Check your PDF files.")
            return
        
        save_documents(documents, args.cache_path)
        print("✅ Embeddings created and cached!\n")
    else:
        print(f"✅ Loaded {len(documents)} documents from cache")
        
        subjects = {}
        for doc in documents:
            subject = doc.get("subject", "Unknown")
            subjects[subject] = subjects.get(subject, 0) + 1
        
        print("\n📚 Documents by subject:")
        for subject, count in sorted(subjects.items()):
            print(f"  • {subject}: {count} chunks")
        print()
    
    # Start Q&A loop
    print("=" * 70)
    print("🎓 SYLLABUS RAG SYSTEM")
    print("=" * 70)
    print(f"📊 Model Type: {model_type}")
    print(f"📊 Mode: {'HYBRID (Local + OpenAI)' if USE_HYBRID else 'LOCAL ONLY'}")
    print(f"📚 Total Documents: {len(documents)}")
    print("\n💡 Try asking questions like:")
    print("  • What are the prerequisites for machine learning?")
    print("  • Which courses cover neural networks?")
    print("  • What are the grading policies?")
    print("\nType 'quit' or 'exit' to end session")
    print("=" * 70 + "\n")
    
    while True:
        q = input("❓ Question: ").strip()
        
        if q.lower() in {"quit", "exit", "q"}:
            print("\n👋 Goodbye!")
            break
        
        if not q:
            continue
        
        print("\n🔍 Searching and generating answer...")
        try:
            answer, sources = answer_question(q, documents, model)
            
            print(f"\n💡 Answer:")
            print("-" * 70)
            print(answer)
            print("-" * 70)
            print(f"\n📚 Sources:")
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source}")
            print("\n" + "=" * 70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error processing question: {e}\n")

if __name__ == "__main__":
    main()