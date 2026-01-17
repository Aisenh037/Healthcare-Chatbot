"""
DEMO 3: Complete RAG Pipeline
==============================

WHAT YOU'LL LEARN:
- How Retrieval-Augmented Generation works end-to-end
- How retrieved context improves LLM answers
- Why RAG is better than pure LLM

RUN THIS: python demos/03_rag_pipeline_demo.py

NOTE: This demo simulates LLM generation. In the real app, we use Groq API.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict

print("="*70)
print(" "*15 + "COMPLETE RAG PIPELINE DEMO")
print("="*70)

# Step 1: Document Database
print("\n📚 STEP 1: Medical Knowledge Base...")

knowledge_base = [
    {
        "doc_id": "diabetes_symptoms",
        "content": "Type 2 diabetes symptoms include increased thirst, frequent urination, increased hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections."
    },
    {
        "doc_id": "diabetes_causes",
        "content": "Type 2 diabetes is caused by insulin resistance. Risk factors include obesity, family history, sedentary lifestyle, age over 45, and certain ethnicities."
    },
    {
        "doc_id": "diabetes_treatment",
        "content": "Diabetes management involves blood sugar monitoring, healthy eating, regular exercise, and medications like metformin or insulin therapy when needed."
    },
    {
        "doc_id": "hypertension_info",
        "content": "High blood pressure (hypertension) is when blood pressure readings consistently exceed 140/90 mmHg. It increases risk of heart disease and stroke."
    },
]

print(f"   Knowledge base: {len(knowledge_base)} documents")

# Step 2: Create Vector Index
print("\n🔢 STEP 2: Creating searchable vector index...")

model = SentenceTransformer('all-MiniLM-L6-v2')

vector_index = []
for doc in knowledge_base:
    embedding = model.encode(doc["content"])
    vector_index.append({
        **doc,
        "embedding": embedding
    })

print(f"   ✅ Indexed {len(vector_index)} documents")

# Step 3: Retrieval Function
def retrieve_context(query: str, top_k: int = 2) -> List[Dict]:
    """
    Retrieve most relevant documents for the query
    
    This is the 'R' in RAG (Retrieval)
    """
    query_embedding = model.encode(query)
    
    results = []
    for doc in vector_index:
        similarity = np.dot(query_embedding, doc["embedding"]) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc["embedding"])
        )
        results.append((doc, similarity))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in results[:top_k]]

# Step 4: Generation Function (Simulated)
def generate_answer(query: str, context_docs: List[Dict]) -> str:
    """
    Generate answer using retrieved context
    
    This is the 'AG' in RAG (Augmented Generation)
    
    In real app, this calls Groq LLM API.
    For demo, we simulate based on context.
    """
    # Build context string
    context = "\n\n".join([f"[{doc['doc_id']}]: {doc['content']}" 
                          for doc in context_docs])
    
    # Simulated prompt (what we send to real LLM)
    prompt = f"""You are a medical assistant. Answer the question using ONLY the provided context.

CONTEXT:
{context}

QUESTION: {query}

ANSWER (be concise and cite sources):"""
    
    # Simulated answer (in real app, LLM generates this)
    # We'll just return context for demo
    return f"""Based on the retrieved documents:

{context}

💡 In the real app, Groq LLM would read this context and generate a natural, conversational answer!"""

# Step 5: Complete RAG Pipeline
def rag_pipeline(query: str) -> Dict:
    """
    Complete RAG flow: Retrieve → Augment → Generate
    """
    print(f"\n{'='*70}")
    print(f"RAG PIPELINE FOR: '{query}'")
    print(f"{'='*70}")
    
    # STEP 1: RETRIEVAL
    print("\n📝 STEP 1: RETRIEVAL")
    print("   Searching knowledge base...")
    
    retrieved_docs = retrieve_context(query, top_k=2)
    
    print(f"   ✅ Retrieved {len(retrieved_docs)} relevant documents:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"      {i}. {doc['doc_id']}")
    
    # STEP 2: AUGMENTATION
    print("\n🔗 STEP 2: AUGMENTATION")
    print("   Combining question + retrieved context...")
    
    context_text = "\n".join([doc['content'] for doc in retrieved_docs])
    print(f"   ✅ Context length: {len(context_text)} characters")
    
    # STEP 3: GENERATION
    print("\n🤖 STEP 3: GENERATION")
    print("   Sending to LLM (simulated)...")
    
    answer = generate_answer(query, retrieved_docs)
    
    print(f"   ✅ Generated answer!")
    
    return {
        "query": query,
        "retrieved_docs": retrieved_docs,
        "answer": answer
    }

# Step 6: Run examples
print("\n\n" + "="*70)
print("EXAMPLE RAG QUERIES")
print("="*70)

queries = [
    "What are the symptoms of diabetes?",
    "How is diabetes treated?",
]

for query in queries:
    result = rag_pipeline(query)
    
    print(f"\n📋 FINAL ANSWER:")
    print(result["answer"])
    print()

# Step 7: Compare RAG vs Pure LLM
print("\n\n💡 RAG VS PURE LLM COMPARISON:")
print("="*70)

test_question = "What are the symptoms of diabetes?"

print(f"\nQuestion: '{test_question}'")

print(f"\n🔴 PURE LLM (no context):")
print("""
   Problem: LLM only knows what it was trained on
   - May give generic, outdated information
   - Cannot access your specific documents
   - No source citations
   
   Example answer: "Common symptoms include thirst and fatigue..."
   (Generic, no specifics from YOUR medical guidelines)
""")

print(f"\n🟢 RAG (with context):")
print("""
   Advantage: LLM gets YOUR documents as context
   - Answers grounded in your specific medical guidelines
   - Can cite source documents
   - Always up-to-date (just upload new PDFs)
   
   Example answer: "According to [diabetes_symptoms], Type 2 diabetes 
   symptoms include increased thirst, frequent urination, increased hunger,
   unexplained weight loss, fatigue, blurred vision..."
   (Specific, cites source, uses YOUR data!)
""")

# Step 8: Key insights
print("\n\n💡 KEY INSIGHTS - WHY RAG WORKS:")
print("="*70)
print("""
1. RETRIEVAL (Find relevant info):
   - User asks question
   - Search your document database using vector similarity
   - Get top K most relevant chunks (e.g., top 3)

2. AUGMENTATION (Give context to LLM):
   - Take user question + retrieved chunks
   - Build prompt: "Answer using these documents: ..."
   - Send to LLM

3. GENERATION (LLM creates answer):
   - LLM reads context (your documents)
   - Generates answer grounded in YOUR data
   - More accurate than pure LLM

INTERVIEW QUESTIONS:

Q: Why use RAG instead of fine-tuning the LLM?
A: "Fine-tuning is expensive ($1000+) and static. With RAG, I can update 
   knowledge by just uploading new PDFs. No retraining needed. Perfect for 
   medical guidelines that change monthly."

Q: What if no relevant documents are found?
A: "I check the similarity score. If top result < 0.6, I respond 'I don't 
   have enough information to answer this question.' This prevents 
   hallucination."

Q: How do you handle long documents?
A: "I chunk documents into 1000-character segments with 100-char overlap. 
   This ensures each chunk fits in the LLM's context window and maintains 
   coherence at chunk boundaries."
""")

print("\n" + "="*70)
print("Try it yourself with the interactive mode!")
print("="*70)

# Interactive RAG
while True:
    user_question = input("\nAsk a medical question (or 'exit'): ")
    if user_question.lower() in ['exit', 'quit', 'q']:
        break
    
    result = rag_pipeline(user_question)
    print(f"\n📝 Answer:\n{result['answer']}\n")
