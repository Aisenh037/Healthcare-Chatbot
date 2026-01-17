"""
SIMPLIFIED DEMO 2: Vector Search (No Dependencies!)
===================================================

Shows how semantic search works using simple Python

WHAT YOU'LL LEARN:
- How to search documents by meaning
- Why it's better than keyword search
- Ranking by relevance

RUN THIS: python demos/simple_02_vector_search.py
"""

import math

print("="*70)
print(" "*15 + "VECTOR SEARCH DEMO (SIMPLIFIED)")
print("="*70)

# Medical document database with pre-computed embeddings
documents = [
    {
        "id": "doc1",
        "title": "Diabetes Symptoms Guide",
        "content": "Type 2 diabetes symptoms include increased thirst, frequent urination, increased hunger, unexplained weight loss, fatigue, blurred vision.",
        # Simplified embedding (normally 384 dims, using 5 for demo)
        "embedding": [0.9, 0.3, 0.1, 0.7, 0.2]
    },
    {
        "id": "doc2",
        "title": "Hypertension Overview",
        "content": "High blood pressure (hypertension) is when blood pressure readings consistently exceed 140/90 mmHg.",
        "embedding": [0.2, 0.8, 0.3, 0.1, 0.6]
    },
    {
        "id": "doc3",
        "title": "Diabetes Treatment Options",
        "content": "Managing diabetes involves blood sugar monitoring, healthy eating, regular exercise, and medications like metformin.",
        "embedding": [0.85, 0.25, 0.15, 0.65, 0.18]
    },
    {
        "id": "doc4",
        "title": "Heart Disease Prevention",
        "content": "Preventing heart disease requires healthy lifestyle: balanced diet, exercise, no smoking, stress management.",
        "embedding": [0.3, 0.7, 0.4, 0.2, 0.5]
    },
    {
        "id": "doc5",
        "title": "Diabetes Early Warning Signs",
        "content": "Early diabetes indicators: excessive thirst, frequent bathroom trips, unusual hunger, blurry vision, slow-healing cuts.",
        "embedding": [0.88, 0.28, 0.12, 0.68, 0.21]
    }
]

print(f"\n📚 STEP 1: DOCUMENT DATABASE")
print(f"   Loaded {len(documents)} medical documents")
print("   Each has content + embedding vector")

# Search function
def cosine_similarity(vec1, vec2):
    """Calculate similarity between vectors"""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a**2 for a in vec1))
    norm2 = math.sqrt(sum(b**2 for b in vec2))
    return dot / (norm1 * norm2)

def search(query_embedding, top_k=3):
    """Search documents using vector similarity"""
    results = []
    for doc in documents:
        score = cosine_similarity(query_embedding, doc["embedding"])
        results.append((doc, score))
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

# Example searches
queries = [
    {
        "question": "What are diabetes symptoms?",
        "embedding": [0.89, 0.29, 0.11, 0.69, 0.19]  # Similar to diabetes docs
    },
    {
        "question": "How to manage high blood sugar",
        "embedding": [0.83, 0.27, 0.14, 0.64, 0.20]  # Treatment-focused
    },
]

print("\n\n🔍 STEP 2: RUNNING SEARCHES")
print("="*70)

for query_data in queries:
    query = query_data["question"]
    query_emb = query_data["embedding"]
    
    print(f'\n\nQuery: "{query}"')
    print("-" * 70)
    
    results = search(query_emb, top_k=3)
    
    print("\n   📊 Top 3 Results:")
    for rank, (doc, score) in enumerate(results, 1):
        # Visual bar
        bar_length = int(score * 20)
        bar = "█" * bar_length
        
        # Color indicator
        if score > 0.9:
            indicator = "🟢"
        elif score > 0.7:
            indicator = "🟡"
        else:
            indicator = "🔴"
        
        print(f'\n   {rank}. {indicator} {doc["title"]}')
        print(f'      Similarity: {score:.3f} {bar}')
        print(f'      Preview: {doc["content"][:80]}...')

# Comparison
print("\n\n💡 KEYWORD VS SEMANTIC SEARCH")
print("="*70)

test_query = "How to manage high blood sugar"

print(f'\nQuery: "{test_query}"')

print("\n🔴 KEYWORD SEARCH would find:")
print("   - Only docs with words 'blood' AND 'sugar'")
print("   - Misses: 'diabetes' docs (synonym)")  
print("   - Misses: 'treatment' docs (related concept)")  
print("   - Result: Maybe 1 document")

print("\n🟢 SEMANTIC SEARCH (our method) finds:")
test_emb = [0.83, 0.27, 0.14, 0.64, 0.20]
results = search(test_emb, top_k=3)
print(f"   - {len(results)} relevant documents:")
for rank, (doc, score) in enumerate(results, 1):
    print(f"     {rank}. {doc['title']} ({score:.2f})")
print("   - Understands 'diabetes' = 'high blood sugar'")
print("   - Finds treatment/management docs")

print("\n\n💡 KEY INSIGHTS:")
print("="*70)
print("""
1. SEMANTIC RANKING:
   - Documents sorted by similarity score
   - Most relevant appears first
   - Automatic relevance ordering!

2. SYNONYM UNDERSTANDING:
   - "high blood sugar" → finds "diabetes" docs
   - System learns relationships during training
   - No manual synonym dictionary needed

3. REAL USE CASE:
   When user asks: "How to manage high blood sugar"
   
   Step 1: Convert question to embedding
   Step 2: Search all document embeddings
   Step 3: Return top K (e.g., top 5)
   Step 4: Send to LLM for answer generation

4. WHY IT WORKS:
   - Similar meanings → similar vectors
   - Cosine similarity finds similar vectors
   - Math replaces manual rules!

INTERVIEW Q&A:
Q: What if two documents have same similarity score?
A: "We can add tie-breakers like recency, document popularity,
    or user feedback. For medical docs, I'd prioritize recent
    guidelines."

Q: How do you handle typos?
A: "Embeddings are somewhat typo-tolerant because they look at
    overall meaning, not exact characters. But for critical
    medical terms, I'd add spell-check preprocessing."
""")

print("\n" + "="*70)
print("✅ VECTOR SEARCH DEMO COMPLETE!")
print("="*70)
print("\nNext: Run simple_03_rag.py to see the complete RAG pipeline!")
