# Demo Scripts - Learn by Running

These interactive demos show you HOW each component works before you build the full system.

## 🎯 Learning Path

Run demos in order:

### 1️⃣ Embeddings Demo
**What**: Convert text to numbers (vectors)  
**Why**: Foundation of semantic search  
**Run**: `python demos/01_embeddings_demo.py`

**What you'll see**:
- How "diabetes symptoms" and "diabetes warning signs" get similar embeddings
- How similarity scores work (0.0 = unrelated, 1.0 = identical)
- Why embeddings beat keyword matching

---

### 2️⃣ Vector Search Demo
**What**: Search documents using semantic similarity  
**Why**: Find relevant content without exact keyword matches  
**Run**: `python demos/02_vector_search_demo.py`

**What you'll see**:
- Search "high blood sugar" and find "diabetes" documents
- Interactive mode - ask your own questions!
- Comparison: keyword search vs semantic search

---

### 3️⃣ Complete RAG Pipeline Demo
**What**: Full Retrieval-Augmented Generation flow  
**Why**: See how all pieces fit together  
**Run**: `python demos/03_rag_pipeline_demo.py`

**What you'll see**:
- Retrieve → Augment → Generate workflow
- How retrieved context improves LLM answers
- RAG vs pure LLM comparison

---

## 📦 Setup

Install required packages:
```bash
pip install sentence-transformers numpy
```

## 💡 Key Takeaways

After running all demos, you'll understand:

1. **Embeddings**: Text → vectors that capture meaning
2. **Vector Search**: Find similar documents using cosine similarity
3. **RAG**: Retrieve YOUR documents + LLM = Accurate answers

## 🎓 Next Steps

1. Run all 3 demos
2. Read the code comments carefully
3. Experiment with different inputs
4. Explain to yourself how it works
5. Then we'll build the real system!

---

**Interview Tip**: "I built these demos first to deeply understand embeddings, vector search, and RAG before implementing the production system. This helped me make informed architectural decisions."
