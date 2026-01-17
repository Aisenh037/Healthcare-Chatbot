"""
DEMO 2: How Vector Search Works
================================

WHAT YOU'LL LEARN:
- How to store embeddings in a searchable index
- How to find similar documents using vector search
- Difference between keyword vs semantic search

RUN THIS: python demos/02_vector_search_demo.py
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple

print("="*70)
print(" "*20 + "VECTOR SEARCH DEMO")
print("="*70)

# Step 1: Prepare our "document database"
print("\n📚 STEP 1: Creating medical document database...")

documents = [
    {
        "id": "doc1",
        "title": "Diabetes Overview",
        "content": "Diabetes is a chronic condition characterized by high blood sugar levels. Common symptoms include increased thirst, frequent urination, and fatigue."
    },
    {
        "id": "doc2",
        "title": "Hypertension Guide",
        "content": "High blood pressure (hypertension) is a condition where blood pressure in arteries is persistently elevated. It often has no symptoms."
    },
    {
        "id": "doc3",
        "title": "Heart Attack Warning Signs",
        "content": "Myocardial infarction (heart attack) occurs when blood flow to the heart is blocked. Warning signs include chest pain and shortness of breath."
    },
    {
        "id": "doc4",
        "title": "Type 2 Diabetes Management",
        "content": "Managing type 2 diabetes involves monitoring blood glucose, maintaining healthy diet, regular exercise, and sometimes medication."
    },
    {
        "id": "doc5",
        "title": "Asthma Treatment",
        "content": "Asthma is a respiratory condition causing airway inflammation. Treatment includes inhalers and avoiding triggers."
    }
]

print(f"   ✅ Loaded {len(documents)} medical documents")

# Step 2: Generate embeddings for all documents
print("\n🔢 STEP 2: Converting documents to embeddings...")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Store embeddings with document metadata
document_index = []
for doc in documents:
    embedding = model.encode(doc["content"])
    document_index.append({
        "id": doc["id"],
        "title": doc["title"],
        "content": doc["content"],
        "embedding": embedding
    })

print(f"   ✅ Generated embeddings for {len(document_index)} documents")
print(f"   Each embedding: {len(document_index[0]['embedding'])} dimensions")

# Step 3: Search function
def vector_search(query: str, top_k: int = 3) -> List[Tuple[dict, float]]:
    """
    Search for documents similar to query using vector similarity
    
    Args:
        query: User's question
        top_k: Number of results to return
        
    Returns:
        List of (document, similarity_score) tuples
    """
    # Convert query to embedding
    query_embedding = model.encode(query)
    
    # Calculate similarity with all documents
    results = []
    for doc in document_index:
        doc_embedding = doc["embedding"]
        
        # Cosine similarity
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        
        results.append((doc, similarity))
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:top_k]

# Step 4: Run example searches
print("\n🔍 STEP 3: Running example searches...")

queries = [
    "What are diabetes symptoms?",
    "How to manage high blood sugar?",
    "Signs of heart problems",
]

for query in queries:
    print(f"\n{'='*70}")
    print(f"Query: '{query}'")
    print(f"{'='*70}")
    
    results = vector_search(query, top_k=3)
    
    print("\n   📊 Top 3 Results:")
    for rank, (doc, score) in enumerate(results, 1):
        # Visual indicator
        if score > 0.6:
            indicator = "🟢"
        elif score > 0.4:
            indicator = "🟡"
        else:
            indicator = "🔴"
        
        print(f"\n   {rank}. {indicator} {doc['title']} (similarity: {score:.3f})")
        print(f"      {doc['content'][:100]}...")

# Step 5: Keyword vs Semantic comparison
print("\n\n💡 KEYWORD VS SEMANTIC SEARCH COMPARISON:")
print("="*70)

test_query = "How to manage high blood sugar?"

print(f"\nQuery: '{test_query}'")

print("\n🔴 KEYWORD SEARCH would find:")
print("   - Documents containing 'blood sugar' (only doc4 maybe)")
print("   - MISSES: Docs about 'diabetes' (synonym for high blood sugar)")

print("\n🟢 SEMANTIC SEARCH (our method) finds:")
results = vector_search(test_query, top_k=3)
for rank, (doc, score) in enumerate(results, 1):
    print(f"   {rank}. {doc['title']} ({score:.3f})")
    if "diabetes" in doc['title'].lower() or "diabetes" in doc['content'].lower():
        print(f"      ✅ Found diabetes docs (understands it's related!)")

# Step 6: Key insights
print("\n\n💡 KEY INSIGHTS:")
print("="*70)
print("""
1. SEMANTIC UNDERSTANDING:
   - "high blood sugar" query finds "diabetes" documents
   - System understands they're related even without exact word match

2. RANKING BY RELEVANCE:
   - Results sorted by similarity score
   - Most relevant documents appear first
   
3. REAL-WORLD USE CASE:
   - When user asks question, we:
     a) Convert question to embedding
     b) Search our document database
     c) Return top K most relevant chunks
     d) Send to LLM for answer generation

4. WHY THIS IS POWERFUL:
   - No need to manually tag documents with keywords
   - Automatically understands synonyms and related concepts
   - Works across languages (with multilingual models)

INTERVIEW TIP:
"I used vector search because it captures semantic meaning. When a patient 
asks about 'high blood sugar', the system finds diabetes documents even 
though they don't contain that exact phrase. This is impossible with 
traditional keyword search."
""")

print("\n" + "="*70)
print("EXPERIMENT: Try your own medical queries!")
print("="*70)

# Interactive mode
while True:
    user_query = input("\nEnter your question (or 'exit' to quit): ")
    if user_query.lower() in ['exit', 'quit', 'q']:
        break
    
    results = vector_search(user_query, top_k=3)
    print(f"\n   Top 3 Results for: '{user_query}'")
    for rank, (doc, score) in enumerate(results, 1):
        print(f"   {rank}. {doc['title']} ({score:.3f})")
