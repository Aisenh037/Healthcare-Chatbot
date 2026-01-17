"""
FILE 3: VECTOR STORAGE
======================

CONCEPT: Store embeddings and search by similarity

WHAT THIS FILE DOES:
1. Stores text chunks with their embedding vectors
2. Searches for similar vectors (semantic search)
3. Manages metadata (source file, chunk ID, etc.)

WHY WE NEED THIS:
- Need to store 100s-1000s of document chunks
- Need fast similarity search (find top-K most similar)
- In production: Pinecone (managed, scales to millions)
- For learning: Python list (understand the algorithm!)

INTERVIEW Q&A:
--------------
Q: Why not use a regular database like PostgreSQL?
A: "PostgreSQL is designed for exact matches (WHERE id=5). Vector search
   needs to find 'similar' items, not exact matches. This requires different
   indexing (HNSW, IVF). Pinecone is optimized for vector similarity search."

Q: How does vector search work?
A: "1. Convert query to vector
    2. Calculate similarity with ALL stored vectors (cosine similarity)
    3. Sort by similarity score (highest first)
    4. Return top K results
    
    This naive approach is O(n). Production systems use ANN (Approximate
    Nearest Neighbor) algorithms like HNSW for O(log n) search."

Q: When would you switch from Python list to Pinecone?
A: "Python list works for <10K vectors. Beyond that:
    - Search becomes slow (O(n) linear scan)
    - Memory issues (all vectors in RAM)
    - No persistence (lost on restart)
    
    I'd switch to Pinecone at ~5K vectors or when search latency >100ms."
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Import our embedding generator
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Direct import from file
import importlib.util
spec = importlib.util.spec_from_file_location(
    "embeddings", 
    Path(__file__).parent / "2_create_embeddings.py"
)
embeddings_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embeddings_module)
EmbeddingGenerator = embeddings_module.EmbeddingGenerator


class SimpleVectorStore:
    """
    Minimal vector database using Python list.
    
    Design Decision: Start simple, upgrade later
    WHY:
    - Easy to understand (just a Python list!)
    - Shows the core algorithm (no black box)
    - Good for <10K vectors
    - Can swap to Pinecone later without changing interface
    
    Production Alternative:
    - Pinecone: Managed, scales to 10M+ vectors, $70/month
    - Weaviate: Self-hosted, open-source, more complex setup
    - Chroma: Embedded database, good for 100K vectors
    
    Interview Note:
    "I implemented vector storage as a simple list to understand the algorithm.
     In production, I'd use Pinecone for scalability. But by implementing it
     myself first, I understand exactly what Pinecone is doing under the hood."
    """
    
    def __init__(self, embedding_generator: Optional[EmbeddingGenerator] = None):
        """
        Initialize vector store.
        
        Args:
            embedding_generator: Optional embedder (creates one if not provided)
            
        Data Structure:
        self.vectors = [
            {
                'id': 0,
                'text': "Diabetes symptoms include...",
                'vector': np.array([0.1, 0.2, ...]),  # 384 dims
                'metadata': {'source': 'diabetes.pdf', 'chunk_id': 0}
            },
            ...
        ]
        """
        self.vectors = []
        self.next_id = 0
        
        # Embedding generator
        if embedding_generator is None:
            print("📦 Initializing embedding generator...")
            self.embedder = EmbeddingGenerator()
        else:
            self.embedder = embedding_generator
        
        print("✅ VectorStore initialized")
        print(f"   Storage: Python list (in-memory)")
        print(f"   Embedder: {self.embedder.model}")
    
    
    def add(self, text: str, metadata: Optional[Dict] = None) -> int:
        """
        Add text to vector store.
        
        Args:
            text: Text to store (will be embedded automatically)
            metadata: Optional metadata dict
            
        Returns:
            Assigned ID
            
        Process:
        1. Generate embedding for text
        2. Store: {id, text, vector, metadata}
        3. Increment next_id
        
        Interview Note:
        "I auto-generate embeddings on add() so users don't need to
         manage embeddings separately. In production with Pinecone,
         I'd batch-add for efficiency (100s of vectors at once)."
        """
        # Generate embedding
        vector = self.embedding_generator.embed_text(text)
        
        # Create entry
        entry = {
            'id': len(self.vectors),
            'text': text,
            'vector': vector,
            'metadata': metadata or {},
            'added_at': datetime.utcnow().isoformat()
        }
        
        self.vectors.append(entry)
        
        # Auto-save after write
        self.save(self.persist_path)
        
        return entry['id']
    
    
    def batch_add(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> List[int]:
        """
        Add multiple documents efficiently and auto-save.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dicts (same length as texts)
            
        Returns:
            List of assigned IDs
            
        Optimization:
        Instead of:
            for text in texts:
                add(text)  # 100 separate embed() calls
        
        Do:
            embeddings = batch_embed(texts)  # 1 batched call (10x faster!)
        
        Interview Note:
        "Batching is critical for performance. Embedding 1000 texts one-by-one
         takes ~2 minutes. Batched takes ~12 seconds (10x faster). GPU/CPU can
         process multiple texts in parallel."
        """
        print(f"\n📦 Batch adding {len(texts)} texts...")
        
        # Batch generate embeddings (FAST!)
        vectors = self.embedder.batch_embed(texts)
        
        # Add all to store
        assigned_ids = []
        for i, (text, vector) in enumerate(zip(texts, vectors)):
            metadata = metadatas[i] if metadatas else {}
            
            entry = {
                'id': self.next_id,
                'text': text,
                'vector': vector,
                'metadata': metadata,
                'added_at': datetime.utcnow().isoformat()
            }
            
            self.vectors.append(entry)
            assigned_ids.append(self.next_id)
            self.next_id += 1
        
        print(f"✅ Added {len(texts)} texts (IDs: {assigned_ids[0]}-{assigned_ids[-1]})")
        
        return assigned_ids
    
    
    def search(self, query: str, top_k: int = 5, 
               min_score: float = 0.0) -> List[Tuple[Dict, float]]:
        """
        Search for most similar vectors.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of (entry, score) tuples, sorted by score (highest first)
            
        Algorithm:
        1. Embed query → query_vector
        2. Calculate cosine similarity with ALL vectors
        3. Sort by similarity (descending)
        4. Filter by min_score
        5. Return top K
        
        Complexity: O(n) where n = number of vectors
        
        Interview Note:
        "This is naive search - O(n) linear scan. For small datasets (<10K),
         it's fine and fast enough (<100ms). For production with millions of
         vectors, I'd use Pinecone with HNSW indexing for O(log n) search."
        """
        if not self.vectors:
            return []
        
        # Embed query
        query_vector = self.embedder.embed_text(query)
        
        # Calculate similarities with ALL vectors
        # Optimization: Use batch cosine similarity (matrix ops)
        all_vectors = np.array([v['vector'] for v in self.vectors])
        similarities = self.embedder.batch_cosine_similarity(query_vector, all_vectors)
        
        # Combine with entries
        results = [(self.vectors[i], float(sim)) 
                   for i, sim in enumerate(similarities)]
        
        # Filter by min_score
        if min_score > 0:
            results = [(entry, score) for entry, score in results 
                      if score >= min_score]
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top K
        return results[:top_k]
    
    
    def get_by_id(self, vector_id: int) -> Optional[Dict]:
        """Get vector entry by ID"""
        for entry in self.vectors:
            if entry['id'] == vector_id:
                return entry
        return None
    
    
    def delete_by_id(self, vector_id: int) -> bool:
        """Delete vector by ID"""
        for i, entry in enumerate(self.vectors):
            if entry['id'] == vector_id:
                del self.vectors[i]
                return True
        return False
    
    
    def count(self) -> int:
        """Get total number of vectors"""
        return len(self.vectors)
    
    
    def get_stats(self) -> Dict:
        """Get store statistics"""
        return {
            'total_vectors': len(self.vectors),
            'embedding_dimension': self.embedder.get_embedding_dimension(),
            'next_id': self.next_id,
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    
    def _estimate_memory_usage(self) -> float:
        """
        Estimate memory usage in MB.
        
        Calculation:
        - Each vector: 384 floats × 4 bytes = 1.5 KB
        - Each text: ~500 chars × 1 byte = 0.5 KB
        - Total per entry: ~2 KB
        - 1000 entries: ~2 MB
        """
        if not self.vectors:
            return 0.0
        
        # Bytes per vector (384 floats, each 4 bytes)
        vector_bytes = 384 * 4
        
        # Average text bytes
        avg_text_bytes = np.mean([len(v['text'].encode()) for v in self.vectors[:min(100, len(self.vectors))]])
        
        # Total per entry
        bytes_per_entry = vector_bytes + avg_text_bytes + 200  # +200 for metadata/overhead
        
        total_bytes = bytes_per_entry * len(self.vectors)
        total_mb = total_bytes / (1024 * 1024)
        
        return round(total_mb, 2)
    
    
    def save(self, filepath: str):
        """
        Save vector store to JSON file.
        
        Args:
            filepath: Path to save file
            
        Note: Vectors are converted to lists for JSON serialization
        
        Interview Note:
        "For production, I'd use a proper database (Pinecone persists automatically).
         This JSON save is just for local development/testing."
        """
        print(f"\n💾 Saving vector store to {filepath}...")
        
        # Convert numpy arrays to lists for JSON
        data = {
            'vectors': [
                {
                    'id': v['id'],
                    'text': v['text'],
                    'vector': v['vector'].tolist(),  # numpy → list
                    'metadata': v['metadata'],
                    'added_at': v['added_at']
                }
                for v in self.vectors
            ],
            'next_id': self.next_id,
            'saved_at': datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Saved {len(self.vectors)} vectors")
    
    
    def load(self, filepath: str):
        """
        Load vector store from JSON file.
        
        Args:
            filepath: Path to load from
        """
        print(f"\n📂 Loading vector store from {filepath}...")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        self.vectors = [
            {
                'id': v['id'],
                'text': v['text'],
                'vector': np.array(v['vector']),  # list → numpy
                'metadata': v['metadata'],
                'added_at': v['added_at']
            }
            for v in data['vectors']
        ]
        
        self.next_id = data['next_id']
        
        print(f"✅ Loaded {len(self.vectors)} vectors")


# ==============================================================================
# DEMO & TESTING
# ==============================================================================

def demo_vector_store():
    """
    Demo script showing how vector storage works.
    
    Run this to test: python minimal_rag/3_store_vectors.py
    """
    print("="*70)
    print(" "*20 + "VECTOR STORAGE DEMO")
    print("="*70)
    
    # Initialize store
    store = SimpleVectorStore()
    
    # Test 1: Add single vectors
    print("\n" + "="*70)
    print("TEST 1: Adding Individual Texts")
    print("="*70)
    
    texts = [
        "Diabetes symptoms include increased thirst and frequent urination",
        "High blood pressure is managed with diet and medication",
        "Type 2 diabetes is treated with lifestyle changes and metformin"
    ]
    
    for i, text in enumerate(texts):
        id = store.add(text, metadata={'source': f'doc_{i}.txt'})
        print(f"  Added ID {id}: {text[:50]}...")
    
    # Test 2: Batch add (faster!)
    print("\n" + "="*70)
    print("TEST 2: Batch Adding (More Efficient)")
    print("="*70)
    
    batch_texts = [
        "Diabetes early warning signs: excessive thirst, hunger, fatigue",
        "Hypertension treatment involves salt reduction and exercise",
        "Managing blood sugar through diet and physical activity"
    ]
    
    ids = store.batch_add(batch_texts)
    
    # Test 3: Search
    print("\n" + "="*70)
    print("TEST 3: Semantic Search")
    print("="*70)
    
    queries = [
        "What are diabetes symptoms?",
        "How to manage high blood pressure?"
    ]
    
    for query in queries:
        print(f'\n🔍 Query: "{query}"')
        results = store.search(query, top_k=3, min_score=0.3)
        
        print(f"   Found {len(results)} results:\n")
        for i, (entry, score) in enumerate(results, 1):
            indicator = "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🟠"
            print(f"   {i}. {indicator} Score: {score:.3f}")
            print(f"      Text: {entry['text'][:60]}...")
            print(f"      Metadata: {entry['metadata']}")
            print()
    
    # Test 4: Stats
    print("="*70)
    print("TEST 4: Store Statistics")
    print("="*70)
    
    stats = store.get_stats()
    print(f"\n📊 Vector Store Stats:")
    print(f"   Total vectors: {stats['total_vectors']}")
    print(f"   Embedding dimension: {stats['embedding_dimension']}")
    print(f"   Memory usage: {stats['memory_usage_mb']} MB")
    
    # Test 5: Save/Load
    print("\n" + "="*70)
    print("TEST 5: Persistence (Save/Load)")
    print("="*70)
    
    save_path = "test_vector_store.json"
    store.save(save_path)
    
    # Load into new store
    new_store = SimpleVectorStore()
    new_store.load(save_path)
    
    print(f"\n✅ Loaded store has {new_store.count()} vectors")
    
    # Clean up
    Path(save_path).unlink()
    print(f"   Cleaned up {save_path}")
    
    # Key insights
    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("""
    1. ✅ Vector store manages text + embeddings + metadata
    2. ✅ Semantic search finds similar items (not exact matches)
    3. ✅ Batch operations are 10x faster than individual adds
    4. ✅ Python list works for <10K vectors
    5. ✅ For production: Switch to Pinecone (scales to millions)
    
    In Interview:
    "I built a simple vector store using Python list to understand the algorithm.
     It stores text chunks with their 384-dim embeddings. Search works by:
     1) Embedding the query
     2) Calculating cosine similarity with all stored vectors
     3) Returning top K most similar
     
     This is O(n) linear scan. For <10K vectors, it's fast enough (<100ms).
     For production, I'd use Pinecone which uses HNSW indexing for O(log n)
     search - 100x faster for millions of vectors."
    """)


if __name__ == "__main__":
    demo_vector_store()
