# Quick Start Guide - Run the Demos

## 🚀 Step 1: Install Dependencies

The demos need `sentence-transformers` package. Install it:

```bash
pip install sentence-transformers numpy
```

**Note**: First run downloads the model (~80MB). Internet required!

---

## 🎯 Step 2: Run Demos in Order

### Demo 1: Embeddings (5 minutes)
```bash
python demos/01_embeddings_demo.py
```

**You'll learn**:
- How text becomes numbers (vectors)
- Why "diabetes symptoms" and "warning signs" are similar
- Cosine similarity calculations

**Expected output**: See similarity scores showing semantic understanding!

---

### Demo 2: Vector Search (7 minutes)
```bash
python demos/02_vector_search_demo.py
```

**You'll learn**:
- How to search documents without keywords
- Why vector search beats keyword matching
- Interactive mode - ask your own questions!

**Try**: Type "high blood sugar" and watch it find "diabetes" docs!

---

### Demo 3: Complete RAG (10 minutes)
```bash
python demos/03_rag_pipeline_demo.py
```

**You'll learn**:
- Full RAG workflow: Retrieve → Augment → Generate
- How context improves LLM answers
- RAG vs pure LLM comparison

**Try**: Ask medical questions and see relevant docs retrieved!

---

## 📚 Step 3: Read the Code

After running, open each demo file and read the comments:

1. **Understand the WHY**: Each section explains business justification
2. **Learn the HOW**: Step-by-step implementation details  
3. **Prepare for Interviews**: "Interview Q&A" sections show common questions

---

## 💡 Step 4: Experiment!

Modify the demos:
- Change the sentences in Demo 1
- Add your own medical documents in Demo 2
- Try different queries in Demo 3

**Learning happens by doing!**

---

## 🎯 Next Steps (After Demos)

Once you understand how each component works:

1. **Option 2**: Deep-dive into concepts (read LEARNING.md in detail)
2. **Option 1**: Start building the MVP (we'll code together)

---

## ⚠️ Troubleshooting

**Error: "No module named sentence_transformers"**
```bash
pip install sentence-transformers
```

**First run is slow**
- Model downloads on first run (~80MB)
- Subsequent runs are fast!

**Import errors**
```bash
pip install numpy sentence-transformers
```

---

## 🎓 What You'll Gain

After running all 3 demos, you can explain to an interviewer:

✅ "How embeddings capture semantic meaning"  
✅ "Why vector search beats keyword search"  
✅ "How RAG combines retrieval with LLM generation"  
✅ "Trade-offs between different similarity metrics"

**Interview-ready in 30 minutes!** 🚀
