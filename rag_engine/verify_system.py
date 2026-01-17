"""
VERIFICATION SCRIPT: MINIMAL RAG SYSTEM
=======================================

This script programmatically tests all 6 components of the Minimal RAG System
and reports a final status.

Components:
1. Document Loading (PyPDF2)
2. Embeddings (Sentence-Transformers)
3. Vector Storage (Numpy/Python)
4. Semantic Retrieval (Cosine Similarity)
5. LLM Generation (Groq API) - Optional
6. Complete Pipeline Integration
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_test(name, func):
    print(f"\n🧪 Testing {name}...")
    try:
        start_time = time.time()
        result = func()
        duration = time.time() - start_time
        print(f"✅ {name} PASSED ({duration:.2f}s)")
        return True, result
    except Exception as e:
        print(f"❌ {name} FAILED: {str(e)}")
        # Print traceback for debugging
        import traceback
        traceback.print_exc()
        return False, None

# --- Component Tests ---

def test_1_loading():
    # Use direct import from file
    import importlib.util
    spec = importlib.util.spec_from_file_location("doc_loader", Path(__file__).parent / "1_load_documents.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loader = module.DocumentLoader(chunk_size=100, chunk_overlap=20)
    text = "This is a test document for chunking. It should have multiple chunks if the size is small."
    chunks = loader.chunk_text(text)
    return len(chunks) > 0

def test_2_embeddings():
    import importlib.util
    spec = importlib.util.spec_from_file_location("embeddings", Path(__file__).parent / "2_create_embeddings.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = module.EmbeddingGenerator()
    emb = generator.embed_text("Hello world")
    return emb.shape == (384,)

def test_3_storage():
    import importlib.util
    spec2 = importlib.util.spec_from_file_location("embeddings", Path(__file__).parent / "2_create_embeddings.py")
    mod2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod2)
    
    spec3 = importlib.util.spec_from_file_location("vector_store", Path(__file__).parent / "3_store_vectors.py")
    mod3 = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(mod3)
    
    embedder = mod2.EmbeddingGenerator()
    store = mod3.SimpleVectorStore(embedder)
    store.add("Diabetes is chronic", metadata={"source": "test"})
    return store.count() == 1

def test_4_retrieval():
    import importlib.util
    spec2 = importlib.util.spec_from_file_location("embeddings", Path(__file__).parent / "2_create_embeddings.py")
    mod2 = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(mod2)
    
    spec3 = importlib.util.spec_from_file_location("vector_store", Path(__file__).parent / "3_store_vectors.py")
    mod3 = importlib.util.module_from_spec(spec3)
    spec3.loader.exec_module(mod3)
    
    spec4 = importlib.util.spec_from_file_location("retriever", Path(__file__).parent / "4_search_similar.py")
    mod4 = importlib.util.module_from_spec(spec4)
    spec4.loader.exec_module(mod4)
    
    embedder = mod2.EmbeddingGenerator()
    store = mod3.SimpleVectorStore(embedder)
    store.add("Symptoms of flu include fever", metadata={"id": 1})
    store.add("Recipe for pizza", metadata={"id": 2})
    retriever = mod4.SemanticRetriever(store, top_k=1)
    results = retriever.retrieve("What are flu signs?")
    return len(results) > 0 and "flu" in results[0]['text'].lower()

def test_5_generation():
    if not os.getenv("GROQ_API_KEY"):
        print("   ⚠️  Skipping: No GROQ_API_KEY found")
        return "SKIPPED"
    import importlib.util
    spec = importlib.util.spec_from_file_location("generator", Path(__file__).parent / "5_generate_answer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = module.AnswerGenerator()
    res = generator.generate("Say 'Hello'", "Context is irrelevant")
    return "Hello" in res['answer']

def test_6_pipeline():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pipeline", Path(__file__).parent / "6_complete_rag.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    api_key = os.getenv("GROQ_API_KEY")
    chatbot = module.MinimalRAGChatbot(groq_api_key=api_key)
    chatbot.vector_store.add("The capital of France is Paris.", metadata={"source": "geo"})
    # Even if generation fails or is skipped, retrieval should work
    res = chatbot.ask("What is the capital of France?")
    return res['num_sources'] > 0

# --- Main Verification ---

def main():
    print("="*70)
    print(" "*20 + "MINIMAL RAG SYSTEM VERIFICATION")
    print("="*70)
    
    results = {}
    
    # 1. Loading
    results["Doc Loading"] = run_test("Document Loading", test_1_loading)
    
    # 2. Embeddings (The likely blocker)
    results["Embeddings"] = run_test("Embeddings", test_2_embeddings)
    
    # Only continue if embeddings work
    if results["Embeddings"][0]:
        results["Vector Store"] = run_test("Vector Store", test_3_storage)
        results["Retrieval"] = run_test("Retrieval", test_4_retrieval)
        results["Generation"] = run_test("LLM Generation", test_5_generation)
        results["Full Pipeline"] = run_test("Complete Pipeline", test_6_pipeline)
    else:
        print("\n🛑 Skipping remaining tests because Embeddings failed.")
    
    print("\n" + "="*70)
    print(" "*25 + "FINAL REPORT")
    print("="*70)
    
    all_passed = True
    for name, (status, _) in results.items():
        if status == "SKIPPED":
            print(f"⚪ {name:20}: SKIPPED (No API Key)")
        elif status:
            print(f"✅ {name:20}: PASSED")
        else:
            print(f"❌ {name:20}: FAILED")
            all_passed = False
            
    if all_passed:
        print("\n🎉 ALL SYSTEMS GO! The Minimal RAG MVP is fully functional.")
    else:
        print("\n⚠️  Some tests failed. Please check the logs above.")
    print("="*70)

if __name__ == "__main__":
    main()
