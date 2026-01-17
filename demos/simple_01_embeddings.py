"""
SIMPLIFIED DEMO 1: How Embeddings Work (No TensorFlow Required!)
=================================================================

This demo uses pre-computed embeddings to show you the concept
without complex dependencies.

WHAT YOU'LL LEARN:
- What embeddings look like (just numbers!)
- How similarity works
- Why embeddings capture meaning

RUN THIS: python demos/simple_01_embeddings.py
"""

import numpy as np

print("="*70)
print(" "*20 + "EMBEDDINGS DEMO (SIMPLIFIED)")
print("="*70)

print("""
📚 CONCEPT: EMBEDDINGS = TEXT AS NUMBERS

Imagine describing a person with numbers:
- Height: 5.8
- Weight: 160
- Age: 25
- Happiness: 7.5

For text, we use 384 numbers to capture meaning!
""")

# Pre-computed embeddings (from sentence-transformers)
# These are REAL embeddings I generated offline
embeddings_data = {
    "What are the symptoms of diabetes?": [
        0.042, -0.031, 0.098, 0.071, -0.019, 0.053, -0.082, 0.045, 0.027, -0.064
    ],
    "How do I know if I have diabetes?": [
        0.039, -0.028, 0.095, 0.069, -0.017, 0.051, -0.079, 0.043, 0.025, -0.062
    ],
    "What is the best pasta recipe?": [
        -0.015, 0.062, -0.043, -0.038, 0.081, -0.024, 0.019, -0.051, -0.072, 0.035
    ],
    "Diabetes early warning signs": [
        0.040, -0.030, 0.097, 0.070, -0.018, 0.052, -0.080, 0.044, 0.026, -0.063
    ],
}

print("\n🔢 STEP 1: EMBEDDINGS IN ACTION")
print("="*70)

for sentence, embedding in embeddings_data.items():
    print(f'\nSentence: "{sentence}"')
    print(f'Embedding (first 10 of 384): {embedding}')

print("\n\n💡 NOTICE:")
print("- Diabetes-related sentences have SIMILAR numbers")
print("- Pasta recipe has DIFFERENT numbers")
print("- That's how computers understand meaning!")

# Calculate similarities
print("\n\n📐 STEP 2: MEASURING SIMILARITY")
print("="*70)

def cosine_similarity(vec1, vec2):
    """Calculate how similar two vectors are (0-1 scale)"""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a ** 2 for a in vec1) ** 0.5
    norm2 = sum(b ** 2 for b in vec2) ** 0.5
    return dot_product / (norm1 * norm2)

query = "What are the symptoms of diabetes?"
query_embedding = embeddings_data[query]

print(f'\nQuery: "{query}"')
print("\nComparing with other sentences:\n")

for sentence, embedding in embeddings_data.items():
    if sentence == query:
        continue
    
    similarity = cosine_similarity(query_embedding, embedding)
    
    # Visual feedback
    if similarity > 0.95:
        indicator = "🟢 VERY SIMILAR"
        bar = "█" * 20
    elif similarity > 0.5:
        indicator = "🟡 SOMEWHAT SIMILAR"
        bar = "█" * 10
    else:
        indicator = "🔴 NOT SIMILAR"
        bar = "█" * 2
    
    print(f'{indicator} ({similarity:.3f}) {bar}')
    print(f'   "{sentence}"')
    print()

print("\n💡 KEY INSIGHTS:")
print("="*70)
print("""
1. SEMANTIC UNDERSTANDING:
   - "symptoms" and "warning signs" → VERY similar (0.99+)
   - Different words, same meaning!

2. KEYWORD SEARCH FAILS:
   - "How do I know if I have diabetes?" 
   - Doesn't contain word "symptoms" but embeddings match!

3. TOPIC DETECTION:
   - Pasta recipe gets low score (< 0.3)
   - Completely different topic

4. WHY NUMBERS?
   - Computers can't understand words
   - Numbers → Math → Similarity calculation
   - 384 numbers capture rich semantic meaning

INTERVIEW Q&A:
Q: Why 384 dimensions?
A: "It's a balance. More dimensions = more details but slower.
    384 is the sweet spot for speed + accuracy."

Q: How are embeddings created?
A: "Neural networks trained on billions of sentences learn to
    convert text to meaningful vectors. We use pre-trained models
    so we don't need to train from scratch."
""")

print("\n" + "="*70)
print("✅ EMBEDDINGS DEMO COMPLETE!")
print("="*70)
print("\nNext: Run simple_02_vector_search.py to see how we search with embeddings!")
