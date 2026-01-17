"""
DEMO 1: How Embeddings Work
============================

WHAT YOU'LL LEARN:
- How to convert text to numbers (embeddings)
- How to measure similarity between texts
- Why embeddings are better than keyword matching

RUN THIS: python demos/01_embeddings_demo.py
"""

from sentence_transformers import SentenceTransformer
import numpy as np

print("="*70)
print(" "*20 + "EMBEDDINGS DEMO")
print("="*70)

# Step 1: Load pre-trained model
print("\n📦 STEP 1: Loading embedding model...")
print("   Model: sentence-transformers/all-MiniLM-L6-v2")
print("   Size: 80MB (downloads on first run)")

model = SentenceTransformer('all-MiniLM-L6-v2')
print("   ✅ Model loaded!")

# Step 2: Convert sentences to embeddings
print("\n🔢 STEP 2: Converting sentences to embeddings...")

sentences = [
    "What are the symptoms of diabetes?",
    "How do I know if I have diabetes?",
    "What is the best pasta recipe?",
    "Diabetes early warning signs",
    "Type 2 diabetes indicators"
]

print(f"\n   Converting {len(sentences)} sentences to vectors...")
embeddings = model.encode(sentences)

print(f"   ✅ Generated embeddings!")
print(f"   Shape: {embeddings.shape}")  # (5, 384) - 5 sentences, 384 dimensions each

# Step 3: Examine one embedding
print("\n🔍 STEP 3: Let's look at one embedding...")
print(f'\n   Sentence: "{sentences[0]}"')
print(f"   Embedding (first 10 numbers): {embeddings[0][:10]}")
print(f"   Total dimensions: {len(embeddings[0])}")

# Step 4: Calculate similarities
print("\n📐 STEP 4: Calculating similarity between sentences...")

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Compare first sentence with all others
query_sentence = sentences[0]
query_embedding = embeddings[0]

print(f'\n   Query: "{query_sentence}"')
print("\n   Similarity scores:")

for i, sentence in enumerate(sentences):
    similarity = cosine_similarity(query_embedding, embeddings[i])
    
    # Visual indicator
    if similarity > 0.8:
        indicator = "🟢 Very Similar"
    elif similarity > 0.5:
        indicator = "🟡 Somewhat Similar"
    else:
        indicator = "🔴 Not Similar"
    
    print(f"   {indicator} ({similarity:.3f}) - {sentence}")

# Step 5: Key insights
print("\n💡 KEY INSIGHTS:")
print("="*70)
print("""
1. SEMANTIC UNDERSTANDING:
   - "What are symptoms of diabetes?" and "How do I know if I have diabetes?"
     have HIGH similarity (0.8+) even with different words!
   
2. KEYWORD MATCHING FAILS:
   - Traditional search would miss "How do I know if I have diabetes?"
     because it doesn't contain the word "symptoms"
   
3. TOPIC DETECTION:
   - "pasta recipe" gets LOW similarity (< 0.3) - completely different topic
   
4. SYNONYM RECOGNITION:
   - "symptoms", "warning signs", "indicators" are understood as similar
     even though they're different words

WHY THIS MATTERS FOR RAG:
- When user asks a question, we convert it to an embedding
- Search for similar embeddings in our document database
- Find relevant content even if exact words don't match!
""")

print("\n" + "="*70)
print("EXPERIMENT: Try changing the sentences above and rerun!")
print("="*70)
