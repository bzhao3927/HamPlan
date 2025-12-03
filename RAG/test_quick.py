#!/usr/bin/env python3
"""
Debug script - why aren't department overviews ranking higher?
"""

import pickle
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os

# Load environment
load_dotenv(find_dotenv())
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_openai_embedding(text):
    """Get OpenAI embedding."""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text[:8000]
    )
    return np.array(response.data[0].embedding)

# Load cached documents
cache_file = "cache/complete_system_v6_dept_chunking.pkl"
print(f"Loading from: {cache_file}")
with open(cache_file, "rb") as f:
    documents = pickle.load(f)

print(f"\n{'='*70}")
print(f"Total documents: {len(documents)}")
print(f"  Catalog: {len([d for d in documents if d.get('type') == 'catalog'])}")
print(f"  Department overviews: {len([d for d in documents if d.get('type') == 'department_overview'])}")
print(f"  Syllabi: {len([d for d in documents if d.get('type') == 'syllabus'])}")
print('='*70)

# Show what CS department overview chunks we have
print("\n🎓 CS Department Overview Chunks:")
cs_chunks = [d for d in documents if d.get('type') == 'department_overview' and 'computer-science' in d.get('department', '')]
for i, chunk in enumerate(cs_chunks, 1):
    print(f"{i}. {chunk['source']}")
    print(f"   Preview: {chunk['text'][:150]}...")
    print()

# Test query
query = "What are the requirements for a CS major?"
print(f"\n{'='*70}")
print(f"❓ Query: {query}")
print('='*70)

# Manually calculate scores with boosting
query_emb = get_openai_embedding(query)

# Detect if requirement query
query_lower = query.lower()
requirement_keywords = ['major', 'minor', 'concentration', 'requirement', 'requirements',
                       'required courses', 'what courses', 'courses for', 'need to take']
is_requirement_query = any(keyword in query_lower for keyword in requirement_keywords)

print(f"\n🔍 Is requirement query: {is_requirement_query}")

scores = []
for doc in documents:
    similarity = np.dot(query_emb, doc["embedding"])
    original_similarity = similarity

    # Apply boost
    boost_applied = False
    if is_requirement_query and doc.get("type") == "department_overview":
        similarity *= 2.0
        boost_applied = True

    scores.append({
        "similarity": similarity,
        "original_similarity": original_similarity,
        "boost": boost_applied,
        "doc": doc
    })

# Sort by similarity
scores.sort(key=lambda x: x["similarity"], reverse=True)

# Show top 20
print(f"\n📊 Top 20 Results (with boosting):")
print(f"{'Rank':<5} {'Type':<20} {'Boost':<7} {'Orig':<7} {'Final':<7} {'Source':<60}")
print("-" * 110)

for i, score_data in enumerate(scores[:20], 1):
    doc = score_data["doc"]
    doc_type = doc.get('type', 'unknown')
    source = doc['source'][:55]
    boost_marker = "✓" if score_data["boost"] else ""

    print(f"{i:<5} {doc_type:<20} {boost_marker:<7} {score_data['original_similarity']:<7.4f} {score_data['similarity']:<7.4f} {source}")

# Show detailed view of top 5
print(f"\n{'='*70}")
print("📚 Detailed Top 5:")
print('='*70)
for i, score_data in enumerate(scores[:5], 1):
    doc = score_data["doc"]
    print(f"\n{i}. {doc['source']}")
    print(f"   Type: {doc.get('type')}")
    print(f"   Section: {doc.get('section', 'N/A')}")
    print(f"   Original similarity: {score_data['original_similarity']:.4f}")
    print(f"   Final similarity: {score_data['similarity']:.4f}")
    print(f"   Boosted: {score_data['boost']}")
    print(f"   Text preview:\n   {doc['text'][:300]}")
    print()

# Check: what's the highest-ranking CS dept overview?
print(f"\n{'='*70}")
print("🔍 Where did CS Department Overview chunks rank?")
print('='*70)
for i, score_data in enumerate(scores, 1):
    doc = score_data["doc"]
    if doc.get('type') == 'department_overview' and 'Computer Science' in doc['source']:
        print(f"Rank {i}: {doc['source']}")
        print(f"  Original sim: {score_data['original_similarity']:.4f}")
        print(f"  Boosted sim: {score_data['similarity']:.4f}")
        print(f"  Section: {doc.get('section')}")
        print()
