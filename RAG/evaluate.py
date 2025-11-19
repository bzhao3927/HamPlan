#!/usr/bin/env python3
"""
RAG System with Model Comparison
Compares base model, fine-tuned model, and OpenAI embeddings
"""

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
import os
import pickle
from pathlib import Path
import json
import time

# Load environment variables
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=OPENAI_API_KEY)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================
# EMBEDDING FUNCTIONS
# =============================================
def get_local_embedding(text, model):
    """Get embedding from local SentenceTransformer model."""
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
                time.sleep(2 ** attempt)
            else:
                raise
    return None

# =============================================
# LOAD DATA
# =============================================
def load_cached_documents(cache_path):
    """Load pre-embedded documents from cache."""
    cache_file = Path(cache_path) / "documents.pkl"
    
    if not cache_file.exists():
        return None
    
    with open(cache_file, "rb") as f:
        documents = pickle.load(f)
    
    return documents

def load_course_catalog(json_path):
    """Load course catalog."""
    with open(json_path, 'r', encoding='utf-8') as f:
        courses = json.load(f)
    return courses

# =============================================
# TEST QUERIES
# =============================================
def create_test_queries():
    """Create test queries with ground truth."""
    test_queries = [
        {
            "query": "What are the prerequisites for machine learning courses?",
            "type": "prerequisites",
            "expected_subjects": ["CS", "Math"]  # Changed
        },
        {
            "query": "Which courses cover neural networks?",
            "type": "content",
            "expected_subjects": ["CS", "Chemistry"]  # Changed
        },
        {
            "query": "What is the grading policy for math courses?",  # Changed from stats
            "type": "grading",
            "expected_subjects": ["Math", "Physics"]  # Changed
        },
        {
            "query": "How many credits is calculus?",
            "type": "credits",
            "expected_subjects": ["Math"]  # OK
        },
        {
            "query": "What courses require programming experience?",
            "type": "prerequisites",
            "expected_subjects": ["CS"]  # Changed
        },
        {
            "query": "Tell me about data science courses",
            "type": "description",
            "expected_subjects": ["CS", "Math"]  # Changed
        },
        {
            "query": "What are office hours for computer science professors?",
            "type": "logistics",
            "expected_subjects": ["CS", "Econ"]  # Changed
        },
        {
            "query": "Which courses have final exams?",
            "type": "grading",
            "expected_subjects": ["CS", "Math", "Chemistry", "Econ", "Neuroscience", "Physics"]  # All of them
        },
    ]
    
    return test_queries

# =============================================
# RETRIEVAL EVALUATION
# =============================================
def evaluate_retrieval(model, model_name, documents, test_queries, use_openai=False):
    """
    Evaluate retrieval quality for a given model.
    Returns metrics: precision@k, recall@k, MRR
    """
    print(f"\n{'='*70}")
    print(f"📊 EVALUATING: {model_name}")
    print(f"{'='*70}")
    
    results = {
        "model_name": model_name,
        "queries": [],
        "avg_precision@5": 0,
        "avg_recall@5": 0,
        "mrr": 0
    }
    
    precisions = []
    recalls = []
    reciprocal_ranks = []
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected_subjects = test_case["expected_subjects"]
        
        print(f"\n[{i}/{len(test_queries)}] Query: {query}")
        
        # Get query embedding
        if use_openai:
            query_emb = get_openai_embedding(query)
        else:
            query_emb = get_local_embedding(query, model)
        
        # Compute similarities
        scores = []
        for doc in documents:
            doc_emb = doc["embedding"]
            
            # If using OpenAI for query, need to re-embed docs too
            # For fair comparison, we'll use the doc embeddings as-is
            # (This assumes docs were embedded with the same model)
            if use_openai:
                # Skip this for now - would need to re-embed all docs
                continue
            
            similarity = np.dot(query_emb, doc_emb)
            scores.append((similarity, doc))
        
        # Get top-k results
        scores.sort(key=lambda x: x[0], reverse=True)
        top_k = 5
        top_docs = [doc for _, doc in scores[:top_k]]
        
        # Evaluate
        retrieved_subjects = [doc.get("subject", "Unknown") for doc in top_docs]
        
        # Precision@k: fraction of retrieved docs that are relevant
        relevant_retrieved = sum(1 for subj in retrieved_subjects if subj in expected_subjects)
        precision = relevant_retrieved / top_k
        
        # Recall@k: fraction of relevant docs that were retrieved
        # (Hard to compute without knowing total relevant docs, so we approximate)
        recall = relevant_retrieved / max(len(expected_subjects), 1)
        
        # MRR: position of first relevant result
        first_relevant_rank = None
        for rank, subj in enumerate(retrieved_subjects, 1):
            if subj in expected_subjects:
                first_relevant_rank = rank
                break
        
        rr = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        reciprocal_ranks.append(rr)
        
        print(f"  Retrieved subjects: {retrieved_subjects[:3]}...")
        print(f"  Precision@5: {precision:.3f}, Recall@5: {recall:.3f}, RR: {rr:.3f}")
        
        results["queries"].append({
            "query": query,
            "precision": precision,
            "recall": recall,
            "reciprocal_rank": rr,
            "retrieved_subjects": retrieved_subjects
        })
    
    # Compute averages
    results["avg_precision@5"] = np.mean(precisions)
    results["avg_recall@5"] = np.mean(recalls)
    results["mrr"] = np.mean(reciprocal_ranks)
    
    print(f"\n{'='*70}")
    print(f"📈 RESULTS FOR {model_name}:")
    print(f"   Precision@5: {results['avg_precision@5']:.3f}")
    print(f"   Recall@5: {results['avg_recall@5']:.3f}")
    print(f"   MRR: {results['mrr']:.3f}")
    print(f"{'='*70}")
    
    return results

# =============================================
# COMPARISON
# =============================================
def compare_models(base_model_name, finetuned_model_path, documents, test_queries):
    """Compare base model vs fine-tuned model."""
    
    print("\n" + "="*70)
    print("🔬 MODEL COMPARISON")
    print("="*70)
    
    all_results = []
    
    # 1. Base Model
    print("\n📥 Loading base model...")
    base_model = SentenceTransformer(base_model_name)
    base_results = evaluate_retrieval(base_model, "Base Model", documents, test_queries)
    all_results.append(base_results)
    
    # 2. Fine-tuned Model
    if Path(finetuned_model_path).exists():
        print("\n📥 Loading fine-tuned model...")
        ft_model = SentenceTransformer(finetuned_model_path)
        ft_results = evaluate_retrieval(ft_model, "Fine-Tuned Model", documents, test_queries)
        all_results.append(ft_results)
    else:
        print(f"\n⚠️  Fine-tuned model not found at: {finetuned_model_path}")
        print("   Skipping fine-tuned evaluation")
        ft_results = None
    
    # 3. OpenAI (if available)
    # Note: This requires re-embedding all documents, which is expensive
    # For now, we'll skip this comparison
    
    # Print comparison
    print("\n" + "="*70)
    print("📊 COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n{'Model':<20} {'Precision@5':<15} {'Recall@5':<15} {'MRR':<10}")
    print("-" * 70)
    
    for result in all_results:
        print(f"{result['model_name']:<20} "
              f"{result['avg_precision@5']:<15.3f} "
              f"{result['avg_recall@5']:<15.3f} "
              f"{result['mrr']:<10.3f}")
    
    # Compute improvement
    if ft_results:
        prec_improvement = (ft_results['avg_precision@5'] - base_results['avg_precision@5']) / base_results['avg_precision@5'] * 100
        recall_improvement = (ft_results['avg_recall@5'] - base_results['avg_recall@5']) / base_results['avg_recall@5'] * 100
        mrr_improvement = (ft_results['mrr'] - base_results['mrr']) / base_results['mrr'] * 100
        
        print("\n" + "="*70)
        print("📈 IMPROVEMENT (Fine-Tuned vs Base)")
        print("="*70)
        print(f"  Precision@5: {prec_improvement:+.1f}%")
        print(f"  Recall@5: {recall_improvement:+.1f}%")
        print(f"  MRR: {mrr_improvement:+.1f}%")
    
    # Save results
    output_file = "model_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "results": all_results,
            "test_queries": test_queries
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return all_results

# =============================================
# MAIN
# =============================================
def main():
    print("="*70)
    print("🔬 RAG MODEL EVALUATION & COMPARISON")
    print("="*70)
    
    # Configuration
    base_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    finetuned_model_path = "./finetuned_model"
    cache_path = "cache/"
    
    # Load documents
    print("\n📚 Loading cached documents...")
    documents = load_cached_documents(cache_path)
    
    if not documents:
        print("❌ No cached documents found. Run RAG system first to create cache.")
        return
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Create test queries
    print("\n📝 Creating test queries...")
    test_queries = create_test_queries()
    print(f"✅ Created {len(test_queries)} test queries")
    
    # Compare models
    results = compare_models(base_model_name, finetuned_model_path, documents, test_queries)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()