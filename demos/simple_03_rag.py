"""
SIMPLIFIED DEMO 3: Complete RAG Pipeline (Pure Python!)
=======================================================

Shows how Retrieval-Augmented Generation works end-to-end

WHAT YOU'LL LEARN:
- Complete RAG workflow
- How context improves answers
- RAG vs pure LLM

RUN THIS: python demos/simple_03_rag.py
"""

import math

print("="*70)
print(" "*12 + "COMPLETE RAG PIPELINE DEMO (SIMPLIFIED)")
print("="*70)

# Medical knowledge base
knowledge_base = [
    {
        "doc_id": "diabetes_symptoms",
        "content": "Type 2 diabetes symptoms include increased thirst, frequent urination, increased hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores.",
        "embedding": [0.9, 0.3, 0.1, 0.7, 0.2]
    },
    {
        "doc_id": "diabetes_treatment",
        "content": "Diabetes management involves blood sugar monitoring, healthy eating (low sugar, whole grains), regular exercise 30min/day, and medications like metformin if needed.",
        "embedding": [0.85, 0.25, 0.15, 0.65, 0.18]
    },
    {
        "doc_id": "diabetes_causes",
        "content": "Type 2 diabetes is caused by insulin resistance. Risk factors: obesity, sedentary lifestyle, family history, age over 45.",
        "embedding": [0.82, 0.32, 0.12, 0.68, 0.22]
    },
    {
        "doc_id": "hypertension",
        "content": "Hypertension (high blood pressure) readings above 140/90 mmHg. Managed with diet, exercise, stress reduction, sometimes medication.",
        "embedding": [0.2, 0.8, 0.3, 0.1, 0.6]
    },
]

print(f"\n📚 STEP 1: KNOWLEDGE BASE")
print(f"   {len(knowledge_base)} medical documents available")

# Helper functions
def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a**2 for a in vec1))
    norm2 = math.sqrt(sum(b**2 for b in vec2))
    return dot / (norm1 * norm2)

# RETRIEVAL function
def retrieve(query_embedding, top_k=2):
    """Step 1: Retrieve relevant documents"""
    results = []
    for doc in knowledge_base:
        score = cosine_similarity(query_embedding, doc["embedding"])
        results.append((doc, score))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in results[:top_k]]

# GENERATION function (simulated)
def generate_answer(query, context_docs):
    """
    Step 2: Generate answer using context
    
    NOTE: In real app, this calls LLM API (Groq/OpenAI)
    Here we simulate to show the concept
    """
    # Build context
    context = "\n\n".join([f"[{doc['doc_id']}]:\n{doc['content']}" 
                          for doc in context_docs])
    
    # Simulated LLM prompt (what we'd send to Groq)
    prompt = f"""You are a medical assistant. Answer using ONLY the context.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    
    # Simulated response (LLM would generate this)
    print(f"\n🤖 LLM PROMPT (what we send to Groq):")
    print("-" * 70)
    print(prompt)
    print("-" * 70)
    
    # Simulated answer (manual for demo)
    if "symptom" in query.lower():
        return f"""Based on [{context_docs[0]['doc_id']}]:

Type 2 diabetes symptoms include:
- Increased thirst
- Frequent urination  
- Increased hunger
- Unexplained weight loss
- Fatigue
- Blurred vision
- Slow-healing sores

If you experience these symptoms, consult a healthcare provider.

Source: {context_docs[0]['doc_id']}"""
    
    elif "treat" in query.lower() or "manage" in query.lower():
        return f"""Based on [{context_docs[0]['doc_id']}]:

Diabetes management involves:
1. Blood sugar monitoring (regular checks)
2. Healthy eating (low sugar, whole grains)
3. Regular exercise (30 minutes daily)
4. Medications like metformin (if prescribed)

Always follow your doctor's treatment plan.

Source: {context_docs[0]['doc_id']}"""
    
    else:
        return f"Based on retrieved documents:\n\n{context}"

# RAG Pipeline
def rag_pipeline(query, query_embedding):
    """Complete RAG workflow"""
    print(f"\n\n{'='*70}")
    print(f"RAG PIPELINE: '{query}'")
    print(f"{'='*70}")
    
    # STEP 1: RETRIEVAL
    print("\n📝 STEP 1: RETRIEVAL")
    print("   Searching knowledge base...")
    
    retrieved_docs = retrieve(query_embedding, top_k=2)
    
    print(f"   ✅ Retrieved {len(retrieved_docs)} relevant documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"      {i}. {doc['doc_id']}")
    
    # STEP 2: AUGMENTATION
    print("\n🔗 STEP 2: AUGMENTATION")
    print("   Combining question + context...")
    context_length = sum(len(doc['content']) for doc in retrieved_docs)
    print(f"   ✅ Context: {context_length} characters")
    
    # STEP 3: GENERATION
    print("\n🤖 STEP 3: GENERATION")
    print("   Generating answer with LLM (simulated)...")
    
    answer = generate_answer(query, retrieved_docs)
    
    print(f"\n\n📋 FINAL ANSWER:")
    print("="*70)
    print(answer)
    print("="*70)
    
    return answer

# Run examples
print("\n\n" + "="*70)
print("EXAMPLE 1: Symptom Query")
print("="*70)

query1 = "What are the symptoms of diabetes?"
embedding1 = [0.89, 0.29, 0.1, 0.69, 0.19]  # Similar to diabetes_symptoms
rag_pipeline(query1, embedding1)

print("\n\n" + "="*70)
print("EXAMPLE 2: Treatment Query")
print("="*70)

query2 = "How is diabetes treated?"
embedding2 = [0.84, 0.26, 0.14, 0.66, 0.19]  # Similar to diabetes_treatment
rag_pipeline(query2, embedding2)

# Comparison
print("\n\n\n💡 RAG VS PURE LLM COMPARISON")
print("="*70)

print("""
Question: "What are the symptoms of diabetes?"

🔴 PURE LLM (no retrieval):
   - Uses only pre-trained knowledge
   - May be generic or outdated
   - Cannot cite YOUR hospital's guidelines
   - No source verification
   
   Example: "Common symptoms are thirst and fatigue..."
   (Generic answer from training data)

🟢 RAG (our method):
   1. Retrieve relevant docs from YOUR database
   2. Give docs to LLM as context  
   3. LLM answers using YOUR data
   4. Can cite sources
   
   Example: "According to [diabetes_symptoms]: Type 2 diabetes 
   symptoms include increased thirst, frequent urination..."
   (Specific answer from YOUR medical guidelines!)

WHY RAG WINS:
✅ Answers grounded in YOUR documents
✅ Can cite sources for verification  
✅ Always up-to-date (just upload new PDFs)
✅ No expensive fine-tuning needed
✅ Explainable (show which docs were used)
""")

print("\n💡 KEY INSIGHTS - THE COMPLETE WORKFLOW:")
print("="*70)
print("""
RAG = RETRIEVAL + AUGMENTATION + GENERATION

STEP 1 - RETRIEVAL:
- User asks question
- Convert question to embedding vector
- Search document database using cosine similarity
- Return top K most relevant documents (e.g., top 3)

STEP 2 - AUGMENTATION:
- Take user's question
- Add retrieved documents as "context"
- Build enhanced prompt for LLM
- Example: "Using these documents: [docs], answer: [question]"

STEP 3 - GENERATION:
- Send prompt to LLM (Groq/OpenAI/etc)
- LLM reads context and generates answer
- Answer is grounded in YOUR documents
- Return answer + source citations

INTERVIEW QUESTIONS:

Q: What happens if retrieval finds irrelevant documents?
A: "I check the similarity score. If top result < 0.6 similarity,
    I return 'I don't have enough information.' This prevents the
    LLM from hallucinating based on poor context."

Q: How do you handle very long documents?
A: "I chunk documents into 1000-character segments. Each chunk gets
    embedded separately. When retrieving, I get top K chunks (not
    documents), which gives more precise context."

Q: Why not just fine-tune the LLM on medical data?
A: "Fine-tuning costs $1000+ and creates a static model. With RAG,
    updating knowledge is free - just upload new PDFs. For medical 
    guidelines that change monthly, RAG is more practical."

Q: How would you improve retrieval accuracy?
A: "Three approaches:
    1. Hybrid search (vector + keyword BM25)
    2. Re-ranking with cross-encoder
    3. Query expansion (rewrite question multiple ways)"
""")

print("\n" + "="*70)
print("✅ RAG PIPELINE DEMO COMPLETE!")
print("="*70)
print("""
🎓 WHAT YOU LEARNED:
1. How RAG retrieves relevant documents
2. How context improves LLM answers
3. Why RAG > pure LLM for domain-specific apps
4. Complete workflow: Retrieve → Augment → Generate

🚀 NEXT STEPS:
- Review LEARNING.md for deep concept dive
- Read MVP_PLAN.md to see how we'll build the real app
- Ready to implement this for real!
""")
